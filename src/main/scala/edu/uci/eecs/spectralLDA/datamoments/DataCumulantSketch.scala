package edu.uci.eecs.spectralLDA.datamoments

/**
  * Data Cumulants Calculation.
  */

import edu.uci.eecs.spectralLDA.utils.{RandNLA, TensorOps}
import breeze.linalg._
import breeze.math.Complex
import breeze.numerics.sqrt
import breeze.signal.fourierTr
import breeze.stats.distributions.{Rand, RandBasis}
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import scala.language.postfixOps


/** Sketch of data cumulant
  *
  * Let the truncated eigendecomposition of $M2$ be $U\Sigma U^T$, $M2\in\mathsf{R}^{V\times V}$,
  * $U\in\mathsf{R}^{V\times k}$, $\Sigma\in\mathsf{R}^{k\times k}$, where $V$ is the vocabulary size,
  * $k$ is the number of topics, $k<V$.
  *
  * If we denote $W=U\Sigma^{-1/2}$, then $W^T M2 W\approx I$. We call $W$ the whitening matrix.
  *
  * @param fftSketchWhitenedM3 FFT of the sketch of whitened M3, multiplied by a coefficient
  *                                i.e \frac{(\alpha_0+1)(\alpha_0+2)}{2} M3(W^T,W^T,W^T)
  *                                      = \sum_{i=1}^k\frac{\alpha_i}{\alpha_0}(W^T\mu_i)^{\otimes 3}
  * @param unwhiteningMatrix $(W^T)^{-1}=U\Sigma^{1/2}$
  *
  * REFERENCES
  * [Wang2015] Wang Y et al, Fast and Guaranteed Tensor Decomposition via Sketching, 2015,
  *            http://arxiv.org/abs/1506.04448
  *
  */
case class DataCumulantSketch(fftSketchWhitenedM3: DenseMatrix[Complex], unwhiteningMatrix: DenseMatrix[Double])
  extends Serializable


object DataCumulantSketch {
  def getDataCumulant(dimK: Int,
                      alpha0: Double,
                      documents: RDD[(Long, SparseVector[Double])],
                      sketcher: TensorSketcher[Double, Double],
                      randomisedSVD: Boolean = true)
                     (implicit tolerance: Double = 1e-9, randBasis: RandBasis = Rand)
  : DataCumulantSketch = {
    val sc: SparkContext = documents.sparkContext

    val validDocuments = documents
      .map {
        case (id, wc) => (id, sum(wc), wc)
      }
      .filter {
        case (_, len, _) => len >= 3
      }
    validDocuments.cache()

    val dimVocab = validDocuments.take(1)(0)._3.length
    val numDocs = validDocuments.count()

    println("Start calculating first order moments...")
    val firstOrderMoments: DenseVector[Double] = validDocuments
      .map {
        case (_, length, vec) => vec / length.toDouble
      }
      .reduce(_ + _)
      .map(_ / numDocs.toDouble).toDenseVector
    println("Finished calculating first order moments.")

    println("Start calculating second order moments...")
    val (eigenVectors: DenseMatrix[Double], eigenValues: DenseVector[Double]) = if (randomisedSVD) {
      RandNLA.whiten2(sc, alpha0,
        dimVocab, dimK, numDocs, firstOrderMoments, validDocuments)
    }
    else {
      val E_x1_x2: DenseMatrix[Double] = validDocuments
        .map { case (_, len, vec) =>
          (TensorOps.spVectorTensorProd2d(vec) - diag(vec)) / (len * (len - 1))
        }
        .reduce(_ + _)
        .map(_ / numDocs.toDouble).toDenseMatrix
      val M2: DenseMatrix[Double] = E_x1_x2 - alpha0 / (alpha0 + 1) * (firstOrderMoments * firstOrderMoments.t)

      val eigSym.EigSym(sigma, u) = eigSym((alpha0 + 1) * M2)
      val i = argsort(sigma)
      (u(::, i.slice(dimVocab - dimK, dimVocab)).copy, sigma(i.slice(dimVocab - dimK, dimVocab)).copy)
    }

    println("Finished calculating second order moments and whitening matrix.")

    println("Start whitening data with dimensionality reduction...")
    val W: DenseMatrix[Double] = eigenVectors * diag(eigenValues map { x => 1 / (sqrt(x) + tolerance) })
    println("Finished whitening data.")

    println("Start calculating third order moments...")

    val firstOrderMoments_whitened = W.t * firstOrderMoments

    val broadcasted_W = sc.broadcast(W)
    val broadcasted_sketcher = sc.broadcast(sketcher)

    val fft_Ta1: DenseMatrix[Complex] = validDocuments
      .map {
        case (_, len, vec) => whitenedM3FirstOrderTerms(
          alpha0,
          broadcasted_W.value,
          firstOrderMoments_whitened,
          vec, len,
          broadcasted_sketcher.value
        )
      }
      .reduce(_ + _)
      .map(_ / numDocs.toDouble)

    val fft_Ta2: DenseMatrix[Complex] = validDocuments
      .flatMap {
        case (_, len, vec) => whitenedM3SecondOrderTerms(
          alpha0,
          broadcasted_W.value,
          firstOrderMoments_whitened,
          vec, len
        )
      }
      .reduceByKey(_ + _)
      .map {
        case (i: Int, x: DenseVector[Double]) => fft_sketch2(
          i, x,
          broadcasted_W.value,
          broadcasted_sketcher.value
        )
      }
      .reduce(_ + _)
      .map(_ / numDocs.toDouble)

    val fft_Ta3: DenseMatrix[Complex] = validDocuments
      .flatMap {
        case (_, len, vec) => whitenedM3ThirdOrderTerms(
          alpha0,
          vec, len
        )
      }
      .reduceByKey(_ + _)
      .map {
        case (i: Int, p: Double) => fft_sketch3(
          i, p,
          broadcasted_W.value,
          broadcasted_sketcher.value
        )
      }
      .reduce(_ + _)
      .map(_ / numDocs.toDouble)

    // sketch of q=W^T M1
    val fft_sketch_q_otimes_3 = fft_sketch(
      2 * alpha0 * alpha0 / ((alpha0 + 1) * (alpha0 + 2)),
      Seq(firstOrderMoments_whitened, firstOrderMoments_whitened, firstOrderMoments_whitened),
      sketcher
    )

    broadcasted_W.unpersist()
    broadcasted_sketcher.unpersist()

    // sketch of whitened M3
    val fft_sketch_whitened_M3: DenseMatrix[Complex] = fft_Ta1 + fft_Ta2 + fft_Ta3 + fft_sketch_q_otimes_3

    println("Finished calculating third order moments.")

    val unwhiteningMatrix = eigenVectors * diag(sqrt(eigenValues))

    new DataCumulantSketch(fft_sketch_whitened_M3 * Complex((alpha0 + 1) * (alpha0 + 2) / 2.0, 0), unwhiteningMatrix)
  }


  /** Compute the terms in the contribution of the document to the FFT of the sketch of whitened M3
    *
    * @param alpha0 Topic concentration
    * @param W      Whitening matrix $W\in\mathsf{R^{V\times k}$, where $V$ is the vocabulary size,
    *               $k$ is the reduced dimension, $k<V$
    * @param q      Whitened M1, i.e. $W^T M1$
    * @param n      Word count vector for the current document
    * @param len    Total word counts for the current document
    * @return Sequence of the terms in the contribution of the document to the FFT of the sketch of whitened M3
    *         i.e. $E[x_1\otimes x_2\otimes x_3](W^T,W^T,W^T)-
    *         \frac{\alpha_0}{\alpha_0+2}\left(E[x_1\otimes x_2\otimes M1]
    *         +E[x_1\otimes M1\otimes x_2]
    *         +E[M1\otimes x_1\otimes x_2]\right)(W^T,W^T,W^T)$
    *         Refer to Eq (22) in [Wang2015]
    *
    *         REFERENCES
    *         [Wang2015] Wang Y et al, Fast and Guaranteed Tensor Decomposition via Sketching, 2015,
    *         http://arxiv.org/abs/1506.04448
    *
    */
  private def whitenedM3FirstOrderTerms(alpha0: Double,
                                        W: DenseMatrix[Double],
                                        q: DenseVector[Double],
                                        n: SparseVector[Double],
                                        len: Double,
                                        sketcher: TensorSketcher[Double, Double])
  : DenseMatrix[Complex] = {
    // $p=W^T n$, where n is the original word count vector
    val p: DenseVector[Double] = W.t * n

    val coeff1 = 1.0 / (len * (len - 1) * (len - 2))
    val coeff2 = 1.0 / (len * (len - 1))
    val h1 = alpha0 / (alpha0 + 2)

    val fft_sketch_p: Seq[DenseMatrix[Complex]] = (0 until 3)
      .map { (d) =>
        val sketch: DenseMatrix[Double] = sketcher.sketch(p, d)
        fourierTr(sketch(*, ::))
      }
    val fft_sketch_q: Seq[DenseMatrix[Complex]] = (0 until 3)
      .map { (d) =>
        val sketch: DenseMatrix[Double] = sketcher.sketch(q, d)
        fourierTr(sketch(*, ::))
      }

    val s1 = fft_sketch_p.reduce(_ :* _) * Complex(coeff1, 0) 
    val s2 = (fft_sketch_p(0) :* fft_sketch_p(1) :* fft_sketch_q(2)) * Complex(- coeff2 * h1, 0)
    val s3 = (fft_sketch_p(0) :* fft_sketch_q(1) :* fft_sketch_p(2)) * Complex(- coeff2 * h1, 0)
    val s4 = (fft_sketch_q(0) :* fft_sketch_p(1) :* fft_sketch_p(2)) * Complex(- coeff2 * h1, 0)

    s1 + s2 + s3 + s4
  }

  private def whitenedM3SecondOrderTerms(alpha0: Double,
                                         W: DenseMatrix[Double],
                                         q: DenseVector[Double],
                                         n: SparseVector[Double],
                                         len: Double)
  : Seq[(Int, DenseVector[Double])] = {
    val p: DenseVector[Double] = W.t * n

    val coeff1 = 1.0 / (len * (len - 1) * (len - 2))
    val coeff2 = 1.0 / (len * (len - 1))
    val h1 = alpha0 / (alpha0 + 2)

    var seqTerms = Seq[(Int, DenseVector[Double])]()
    for ((wc_index, wc_value) <- n.activeIterator) {
      seqTerms ++= Seq(
        (wc_index, -coeff1 * wc_value * p),
        (wc_index, coeff2 * h1 * wc_value * q)
      )
    }

    seqTerms
  }

  private def whitenedM3ThirdOrderTerms(alpha0: Double,
                                        n: SparseVector[Double],
                                        len: Double)
  : Seq[(Int, Double)] = {
    val coeff1 = 1.0 / (len * (len - 1) * (len - 2))
    val coeff2 = 1.0 / (len * (len - 1))
    val h1 = alpha0 / (alpha0 + 2)

    var seqTerms = Seq[(Int, Double)]()
    for ((wc_index, wc_value) <- n.activeIterator) {
      seqTerms :+= (wc_index, 2 * coeff1 * wc_value)
    }

    seqTerms
  }


  private def fft_sketch(a: Double, b: Seq[DenseVector[Double]],
                         sketcher: TensorSketcher[Double, Double])
  : DenseMatrix[Complex] = {
    val m: DenseMatrix[Complex] = (0 until 3)
      .map { (d) =>
        val sketch: DenseMatrix[Double] = sketcher.sketch(b(d), d)
        fourierTr(sketch(*, ::))
      }
      .reduce(_ :* _)

    m * Complex(a, 0.0)
  }

  private def fft_sketch2(i: Int,
                          x: DenseVector[Double],
                          W: DenseMatrix[Double],
                          sketcher: TensorSketcher[Double, Double])
  : DenseMatrix[Complex] = {
    val w: DenseVector[Double] = W(i, ::).t
    val fft_sketch_w: Seq[DenseMatrix[Complex]] = (0 until 3)
      .map { (d) =>
        val sketch: DenseMatrix[Double] = sketcher.sketch(w, d)
        fourierTr(sketch(*, ::))
      }
    val fft_sketch_x: Seq[DenseMatrix[Complex]] = (0 until 3)
      .map { (d) =>
        val sketch: DenseMatrix[Double] = sketcher.sketch(x, d)
        fourierTr(sketch(*, ::))
      }

    val prod1 = fft_sketch_w(0) :* fft_sketch_w(1) :* fft_sketch_x(2)
    val prod2 = fft_sketch_w(0) :* fft_sketch_x(1) :* fft_sketch_w(2)
    val prod3 = fft_sketch_x(0) :* fft_sketch_w(1) :* fft_sketch_w(2)

    prod1 + prod2 + prod3
  }

  private def fft_sketch3(i: Int,
                          p: Double,
                          W: DenseMatrix[Double],
                          sketcher: TensorSketcher[Double, Double])
  : DenseMatrix[Complex] = {
    val w: DenseVector[Double] = W(i, ::).t
    val z: DenseMatrix[Complex] = (0 until 3)
      .map { (d) =>
        val sketch: DenseMatrix[Double] = sketcher.sketch(w, d)
        fourierTr(sketch(*, ::))
      }
      .reduce(_ :* _)

    z * Complex(p, 0.0)
  }
}

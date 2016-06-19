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

    val fft_Ta: DenseMatrix[Complex] = validDocuments
      .flatMap {
        case (_, len, vec) => tensorProdTerms(
          alpha0,
          broadcasted_W.value,
          firstOrderMoments_whitened,
          vec, len
        )
      }
      .map {
        case (a: Double, b: Seq[DenseVector[Double]]) => fft_sketch(
          a, b,
          broadcasted_sketcher.value
        )
      }
      .reduce(_ + _)
      .map(_ / numDocs.toDouble)

    broadcasted_W.unpersist()
    broadcasted_sketcher.unpersist()

    // sketch of q=W^T M1
    val fft_sketch_q_otimes_3 = fft_sketch(
      2 * alpha0 * alpha0 / ((alpha0 + 1) * (alpha0 + 2)),
      Seq(firstOrderMoments_whitened, firstOrderMoments_whitened, firstOrderMoments_whitened),
      sketcher
    )

    // sketch of whitened M3
    val fft_sketch_whitened_M3: DenseMatrix[Complex] = fft_Ta + fft_sketch_q_otimes_3

    println("Finished calculating third order moments.")

    val unwhiteningMatrix = eigenVectors * diag(sqrt(eigenValues))

    new DataCumulantSketch(fft_sketch_whitened_M3 * Complex((alpha0 + 1) * (alpha0 + 2) / 2.0, 0), unwhiteningMatrix)
  }



  /** Compute the terms in the contribution of the document to the FFT of the sketch of whitened M3
    *
    * @param alpha0 Topic concentration
    * @param W Whitening matrix $W\in\mathsf{R^{V\times k}$, where $V$ is the vocabulary size,
    *          $k$ is the reduced dimension, $k<V$
    * @param q Whitened M1, i.e. $W^T M1$
    * @param n Word count vector for the current document
    * @param len Total word counts for the current document
    * @return Sequence of the terms in the contribution of the document to the FFT of the sketch of whitened M3
    *         i.e. $E[x_1\otimes x_2\otimes x_3](W^T,W^T,W^T)-
    *                  \frac{\alpha_0}{\alpha_0+2}\left(E[x_1\otimes x_2\otimes M1]
    *                                       +E[x_1\otimes M1\otimes x_2]
    *                                       +E[M1\otimes x_1\otimes x_2]\right)(W^T,W^T,W^T)$
    *         Refer to Eq (22) in [Wang2015]
    *
    * REFERENCES
    * [Wang2015] Wang Y et al, Fast and Guaranteed Tensor Decomposition via Sketching, 2015,
    *            http://arxiv.org/abs/1506.04448
    *
    */
  private def tensorProdTerms(alpha0: Double,
                              W: DenseMatrix[Double],
                              q: DenseVector[Double],
                              n: SparseVector[Double],
                              len: Double)
       : Seq[(Double, Seq[DenseVector[Double]])] = {
    // $p=W^T n$, where n is the original word count vector
    val p: DenseVector[Double] = W.t * n

    val coeff1 = 1.0 / (len * (len - 1) * (len - 2))
    val coeff2 = 1.0 / (len * (len - 1))
    val h1 = alpha0 / (alpha0 + 2)

    var seqTerms = Seq[(Double, Seq[DenseVector[Double]])](
      (coeff1, Seq(p, p, p)),
      (- coeff2 * h1, Seq(p, p, q)),
      (- coeff2 * h1, Seq(p, q, p)),
      (- coeff2 * h1, Seq(q, p, p))
    )

    for ((wc_index, wc_value) <- n.activeIterator) {
      val w: DenseVector[Double] = W(wc_index, ::).t
      seqTerms ++= Seq(
        (- coeff1 * wc_value, Seq(w, w, p)),
        (- coeff1 * wc_value, Seq(w, p, w)),
        (- coeff1 * wc_value, Seq(p, w, w)),
        (2 * coeff1 * wc_value, Seq(w, w, w)),
        (coeff2 * h1 * wc_value, Seq(w, w, q)),
        (coeff2 * h1 * wc_value, Seq(w, q, w)),
        (coeff2 * h1 * wc_value, Seq(q, w, w))
      )
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
      .reduce (_ :* _)

    m * Complex(a, 0.0)
  }
}

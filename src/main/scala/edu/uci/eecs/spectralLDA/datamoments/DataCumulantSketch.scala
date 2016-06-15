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
    val E_x1_x2: DenseMatrix[Double] = validDocuments
      .map { case (_, len, vec) =>
        (TensorOps.spVectorTensorProd2d(vec) - diag(vec)) / (len * (len - 1))
      }
      .reduce(_ + _)
      .map(_ / numDocs.toDouble).toDenseMatrix
    val M2: DenseMatrix[Double] = E_x1_x2 - alpha0 / (alpha0 + 1) * (firstOrderMoments * firstOrderMoments.t)

    val (eigenVectors: DenseMatrix[Double], eigenValues: DenseVector[Double]) = if (randomisedSVD) {
      RandNLA.whiten(sc, alpha0,
        dimVocab, dimK, numDocs, firstOrderMoments, validDocuments)
    }
    else {
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
      .map {
        case (_, len, vec) => update_thirdOrderMoments(
          alpha0,
          broadcasted_W.value,
          firstOrderMoments_whitened,
          vec, len,
          broadcasted_sketcher.value)
      }
      .reduce(_ + _)
      .map(_ / numDocs.toDouble)

    broadcasted_W.unpersist()
    broadcasted_sketcher.unpersist()

    // sketch of q=W^T M1
    val fft_sketch_q: Seq[DenseMatrix[Complex]] = (0 until 3)
      .map { (d) =>
        val sketch: DenseMatrix[Double] = sketcher.sketch(firstOrderMoments_whitened, d)
        fourierTr(sketch(*, ::))
      }
    val fft_sketch_q_otimes_3 = fft_sketch_q(0) :* fft_sketch_q(1) :* fft_sketch_q(2)

    // sketch of whitened M3
    val fft_sketch_whitened_M3: DenseMatrix[Complex] = (fft_Ta
      + fft_sketch_q_otimes_3 * Complex(2 * alpha0 * alpha0 / ((alpha0 + 1) * (alpha0 + 2)), 0)
      )
    println("Finished calculating third order moments.")

    val unwhiteningMatrix = eigenVectors * diag(sqrt(eigenValues))

    new DataCumulantSketch(fft_sketch_whitened_M3 * Complex((alpha0 + 1) * (alpha0 + 2) / 2.0, 0), unwhiteningMatrix)
  }


  /** Compute the contribution of the document to the FFT of the sketch of whitened M3
    *
    * @param alpha0 Topic concentration
    * @param W Whitening matrix $W\in\mathsf{R^{V\times k}$, where $V$ is the vocabulary size,
    *          $k$ is the reduced dimension, $k<V$
    * @param q Whitened M1, i.e. $W^T M1$
    * @param n Word count vector for the current document
    * @param len Total word counts for the current document
    * @param sketcher The sketching facility
    * @return The contribution of the document to the FFT of the sketch of whitened M3
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
  private def update_thirdOrderMoments(alpha0: Double,
                                       W: DenseMatrix[Double],
                                       q: DenseVector[Double],
                                       n: SparseVector[Double],
                                       len: Double,
                                       sketcher: TensorSketcher[Double, Double])
        : DenseMatrix[Complex] = {
    /* ------------------------------------- */

    // $p=W^T n$, where n is the original word count vector
    val p: DenseVector[Double] = W.t * n

    // fft of sketch_p, $p=W^T n$, where $n$ is the original word count vector
    val fft_sketch_p: Seq[DenseMatrix[Complex]] = (0 until 3)
      .map { (d) =>
        val sketch: DenseMatrix[Double] = sketcher.sketch(p, d)
        fourierTr(sketch(*, ::))
      }

    // fft of sketch_q, $q=W^T M1$
    val fft_sketch_q: Seq[DenseMatrix[Complex]] = (0 until 3)
      .map { (d) =>
        val sketch: DenseMatrix[Double] = sketcher.sketch(q, d)
        fourierTr(sketch(*, ::))
      }

    /* ------------------------------------- */

    // sketch of $p^{\otimes 3}$
    val fft_sketch_p_otimes_3 = fft_sketch_p(0) :* fft_sketch_p(1) :* fft_sketch_p(2)

    // sketch of $p\otimes p\otimes q$
    val fft_sketch_p_p_q = fft_sketch_p(0) :* fft_sketch_p(1) :* fft_sketch_q(2)

    // sketch of $p\otimes q\otimes p$
    val fft_sketch_p_q_p = fft_sketch_p(0) :* fft_sketch_q(1) :* fft_sketch_p(2)

    // sketch of $q\otimes p\otimes p$
    val fft_sketch_q_p_p = fft_sketch_q(0) :* fft_sketch_p(1) :* fft_sketch_p(2)

    /* ------------------------------------- */

    // fft of sketch of $\sum_{i=1}^V -n_i\left(w_i\otimes w_i\otimes p+w_i\otimes p\otimes w_i
    //                                           +p\otimes w_i\otimes w_i\right)
    //                +\sum_{i=1}^V 2n_i w_i^{\otimes 3}$
    // ref: Eq (25) in [Wang2015]
    val fft_sum1: DenseMatrix[Complex] = fft_sketch_q(0) * Complex(0, 0)

    // fft of sketch of $\sum_{i=1}^V -n_i\left(w_i\otimes w_i\otimes q
    //                                          +w_i\otimes q\otimes w_i
    //                                          +q\otimes w_i\otimes w_i\right)$
    // ref: Eq (26) in [Wang2015]
    val fft_sum2: DenseMatrix[Complex] = fft_sketch_q(0) * Complex(0, 0)

    for ((wc_index, wc_value) <- n.activeIterator) {
      val fft_sketch_w_i: Seq[DenseMatrix[Complex]] = (0 until 3)
        .map { (d) =>
          val sketch: DenseMatrix[Double] = sketcher.sketch(W(wc_index, ::).t, d)
          fourierTr(sketch(*, ::))
        }

      // fft of sketch of $w_i^{\otimes 3}$
      val fft_sketch_w_i_otimes_3 = fft_sketch_w_i(0) :* fft_sketch_w_i(1) :* fft_sketch_w_i(2)

      // fft of sketch of $w_i\otimes w_i\otimes p
      val fft_sketch_w_i_w_i_p = fft_sketch_w_i(0) :* fft_sketch_w_i(1) :* fft_sketch_p(2)
      // fft of sketch of $w_i\otimes p\otimes w_i
      val fft_sketch_w_i_p_w_i = fft_sketch_w_i(0) :* fft_sketch_p(1) :* fft_sketch_w_i(2)
      // sketch of $p\otimes w_i\otimes w_i$
      val fft_sketch_p_w_i_w_i = fft_sketch_p(0) :* fft_sketch_w_i(1) :* fft_sketch_w_i(2)

      // sketch of $w_i\otimes w_i\otimes q
      val fft_sketch_w_i_w_i_q = fft_sketch_w_i(0) :* fft_sketch_w_i(1) :* fft_sketch_q(2)
      // sketch of $w_i\otimes q\otimes w_i
      val fft_sketch_w_i_q_w_i = fft_sketch_w_i(0) :* fft_sketch_q(1) :* fft_sketch_w_i(2)
      // sketch of $q\otimes w_i\otimes w_i$
      val fft_sketch_q_w_i_w_i = fft_sketch_q(0) :* fft_sketch_w_i(1) :* fft_sketch_w_i(2)

      fft_sum1 :+= - (fft_sketch_w_i_w_i_p + fft_sketch_w_i_p_w_i + fft_sketch_p_w_i_w_i) * Complex(wc_value, 0)
      fft_sum1 :+= fft_sketch_w_i_otimes_3 * Complex(2 * wc_value, 0)

      fft_sum2 :+= - (fft_sketch_w_i_w_i_q + fft_sketch_w_i_q_w_i + fft_sketch_q_w_i_w_i) * Complex(wc_value, 0)
    }

    // sketch of contribution to $E[x_1\otimes x_2\otimes x_3](W^T,W^T,W^T)$
    val fft_sketch_contribution1 = (fft_sketch_p_otimes_3 + fft_sum1) / Complex(len * (len - 1) * (len - 2), 0)

    // sketch of contribution to $\left(E[x_1\otimes x_2\otimes M1]
    //                                  +E[x_1\otimes M1\otimes x_2]
    //                                  +E[M1\otimes x_1\otimes x_2]\right)(W^T,W^T,W^T)$
    val fft_sketch_contribution2 = ((fft_sketch_p_p_q + fft_sketch_p_q_p + fft_sketch_q_p_p + fft_sum2)
      / Complex(len * (len - 1), 0))

    fft_sketch_contribution1 - fft_sketch_contribution2 * Complex(alpha0 / (alpha0 + 2), 0)
  }
}

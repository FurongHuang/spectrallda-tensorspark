package edu.uci.eecs.spectralLDA.utils

import breeze.linalg.eigSym.EigSym
import breeze.linalg.qr.QR
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, argtopk, cholesky, diag, eigSym, inv, qr, svd}
import breeze.numerics.{pow, sqrt}
import breeze.stats.distributions.{Rand, RandBasis}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD


object RandNLA {
  def whiten2(sc: SparkContext,
              alpha0: Double,
              vocabSize: Int, dimK: Int,
              numDocs: Long,
              firstOrderMoments: DenseVector[Double],
              documents: RDD[(Long, Double, SparseVector[Double])])
            (implicit randBasis: RandBasis = Rand)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    assert(vocabSize >= dimK)
    val slackDimK = Math.min(vocabSize - dimK, dimK)

    val para_main: Double = (alpha0 + 1.0) / numDocs.toDouble
    val para_shift: Double = alpha0

    // Multiple shifted M2 by a random Gaussian matrix
    val gaussianRandomMatrix: DenseMatrix[Double] = AlgebraUtil.gaussian(vocabSize, dimK + slackDimK)
    val gaussianRandomMatrix_broadcasted: Broadcast[breeze.linalg.DenseMatrix[Double]] = sc.broadcast(gaussianRandomMatrix)

    val M2_a_S_1_rdd: RDD[(Int, DenseVector[Double])] = documents flatMap {
      this_document => accumulate_M_mul_S(
        gaussianRandomMatrix_broadcasted.value,
        this_document._3, this_document._2)
    } reduceByKey(_ + _)
    val M2_a_S_1: DenseMatrix[Double] = DenseMatrix.zeros[Double](vocabSize, dimK + slackDimK)
    M2_a_S_1_rdd.collect.foreach {
      case (token, v) => M2_a_S_1(token, ::) := v.t
    }

    val M2_a_S: DenseMatrix[Double] = M2_a_S_1 * para_main
          - (firstOrderMoments * (firstOrderMoments.t * gaussianRandomMatrix)) * para_shift
    gaussianRandomMatrix_broadcasted.destroy

    // Obtain the basis of the action space of shifted M2
    val QR(q: DenseMatrix[Double], _) = qr.reduced(M2_a_S)
    val q_broadcasted = sc.broadcast(q)

    // Multiple shifted M2 by the basis of the action space Q
    val M2_a_Q_1_rdd: RDD[(Int, DenseVector[Double])] = documents flatMap {
      this_document => accumulate_M_mul_S(
        q_broadcasted.value,
        this_document._3, this_document._2)
    } reduceByKey(_ + _)
    val M2_a_Q_1: DenseMatrix[Double] = DenseMatrix.zeros[Double](vocabSize, dimK + slackDimK)
    M2_a_Q_1_rdd.collect.foreach {
      case (token, v) => M2_a_Q_1(token, ::) := v.t
    }
    val M2_a_Q = M2_a_Q_1 * para_main
          - (firstOrderMoments * (firstOrderMoments.t * q)) * para_shift

    // Randomised eigendecomposition of M2
    val (s: DenseVector[Double], u: DenseMatrix[Double]) = decomp2(M2_a_Q, q)
    val idx = argtopk(s, dimK)
    val u_M2 = u(::, idx).toDenseMatrix
    val s_M2 = s(idx).toDenseVector

    q_broadcasted.destroy
    (u_M2, s_M2)
  }

  /** Nystrom method for randomised eigendecomposition of Hermitian matrix
    *
    * Note that
    *
    *     A \approx (AQ)(Q^* AQ)^{-1}(AQ)^*
    *
    * We first compute the square root of Q^* AQ=CC^*, then perform SVD on
    * AQ(C^*)^{-1}=USV^{*}. Therefore
    *
    *     A \approx US^2 U^*.
    *
    *
    * @param aq product of the original n-by-n matrix A and a n-by-k test matrix
    * @param q  the n-by-k test matrix
    * @return   the top k eigenvalues, top k eigenvectors of the original matrix A
    */
  def nystrom(aq: DenseMatrix[Double],
              q: DenseMatrix[Double])
      : (DenseVector[Double], DenseMatrix[Double]) = {
    assert(aq.rows == q.rows && aq.cols == q.cols)
    assert(aq.rows >= aq.cols)

    // Q^* AQ
    val qaq = q.t * aq

    // Solve for the squared root of Q^* AQ
    val c = cholesky((qaq + qaq.t) / 2.0)

    // AQC
    val sqrt_ny = aq * inv(c.t)

    // SVD on AQC
    val svd.SVD(u2: DenseMatrix[Double], s2: DenseVector[Double], _) = svd.reduced(sqrt_ny)

    // SVD of A
    (pow(s2, 2.0), u2)
  }

  /** A Nystrom-like method for randomised eigendecomposition of Hermitian matrix
    *
    * We could first do the eigendecomposition (AQ)^* AQ=USU^*. If A=HKH^*; apparently
    * K=S^{1/2}, H^* Q=U^*.
    *
    * From the last equation, AQ=HS^{1/2}H^* Q=HS^{1/2}U^*, therefore
    *
    *    H=(AQ)US^{-1/2}.
    *
    * Empirically for the Spectral LDA model, using the decomposition algorithm often
    * gives better final results than the Nystrom method.
    *
    * @param aq product of the original n-by-n matrix A and a n-by-k test matrix
    * @param q  the n-by-k test matrix
    * @return   the top k eigenvalues, top k eigenvectors of the original matrix A
    */
  def decomp2(aq: DenseMatrix[Double],
              q: DenseMatrix[Double])
      : (DenseVector[Double], DenseMatrix[Double]) = {
    val w = aq.t * aq
    val EigSym(s: DenseVector[Double], u: DenseMatrix[Double]) = eigSym((w + w.t) / 2.0)

    val sqrt_s = sqrt(s)
    val inverse_sqrt_s = sqrt_s map { 1.0 / _ }

    val h: DenseMatrix[Double] = (aq * u) * diag(inverse_sqrt_s)

    (sqrt_s, h)
  }

  /** Return the contribution of a document to M2, multiplied by the test matrix
    *
    * @param S   n-by-k test matrix
    * @param Wc  length-n word count vector
    * @param len total word count
    * @return    M2*S, i.e (Wc*Wc.t-diag(Wc))/(len*(len-1.0))*S
    */
  private[utils] def accumulate_M_mul_S(S: breeze.linalg.DenseMatrix[Double],
                                        Wc: breeze.linalg.SparseVector[Double],
                                        len: Double)
        : Seq[(Int, DenseVector[Double])] = {
    val len_calibrated: Double = math.max(len, 3.0)
    val norm_length: Double = 1.0 / (len_calibrated * (len_calibrated - 1.0))

    val data_mul_S: DenseVector[Double] = S.t * Wc

    Wc.activeIterator.toSeq.map { case (token, count) =>
      (token, (data_mul_S - S(token, ::).t) * count * norm_length)
    }
  }
}

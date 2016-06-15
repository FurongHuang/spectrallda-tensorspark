package edu.uci.eecs.spectralLDA.utils

import breeze.linalg.eigSym.EigSym
import breeze.linalg.qr.QR
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, argtopk, eigSym, qr, svd}
import breeze.stats.distributions.{Rand, RandBasis}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD


object RandNLA {
  def whiten(sc: SparkContext,
                     alpha0: Double,
                     vocabSize: Int, dimK: Int,
                     numDocs: Long,
                     firstOrderMoments: DenseVector[Double],
                     documents: RDD[(Long, Double, SparseVector[Double])])
            (implicit randBasis: RandBasis = Rand)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    val para_main: Double = (alpha0 + 1.0) / numDocs.toDouble
    val para_shift: Double = alpha0

    //val SEED_random: Long = System.currentTimeMillis
    val gaussianRandomMatrix: DenseMatrix[Double] = AlgebraUtil.gaussian(vocabSize, dimK * 2)
    val gaussianRandomMatrix_broadcasted: Broadcast[breeze.linalg.DenseMatrix[Double]] = sc.broadcast(gaussianRandomMatrix)
    val firstOrderMoments_broadcasted: Broadcast[breeze.linalg.DenseVector[Double]] = sc.broadcast(firstOrderMoments.toDenseVector)

    val M2_a_S: DenseMatrix[Double] = documents map {
      this_document => accumulate_M_mul_S(
        vocabSize, dimK * 2,
        alpha0,
        firstOrderMoments_broadcasted.value,
        gaussianRandomMatrix_broadcasted.value,
        this_document._3, this_document._2)
    } reduce(_ + _)

    M2_a_S :*= para_main
    val shiftedMatrix: breeze.linalg.DenseMatrix[Double] = firstOrderMoments * (firstOrderMoments.t * gaussianRandomMatrix)
    M2_a_S -= shiftedMatrix :* para_shift

    val Q = AlgebraUtil.orthogonalizeMatCols(M2_a_S)

    val M2_a_Q: DenseMatrix[Double] = documents map {
      this_document => accumulate_M_mul_S(
        vocabSize,
        dimK * 2, alpha0,
        firstOrderMoments_broadcasted.value,
        Q,
        this_document._3, this_document._2)
    } reduce(_ + _)
    M2_a_Q :*= para_main
    val shiftedMatrix2: breeze.linalg.DenseMatrix[Double] = firstOrderMoments * (firstOrderMoments.t * Q)
    M2_a_Q -= shiftedMatrix2 :* para_shift

    // Note: eigenvectors * Diag(eigenvalues) = M2_a_Q
    val svd.SVD(u: breeze.linalg.DenseMatrix[Double], s: breeze.linalg.DenseVector[Double], v: breeze.linalg.DenseMatrix[Double]) = svd(M2_a_Q.t * M2_a_Q)
    val eigenVectors: DenseMatrix[Double] = (M2_a_Q * u) * breeze.linalg.diag(s.map(entry => 1.0 / math.sqrt(entry)))
    val eigenValues: DenseVector[Double] = s.map(entry => math.sqrt(entry))
    (eigenVectors(::, 0 until dimK), eigenValues(0 until dimK))
  }

  def whiten2(sc: SparkContext,
              alpha0: Double,
              vocabSize: Int, dimK: Int,
              numDocs: Long,
              firstOrderMoments: DenseVector[Double],
              documents: RDD[(Long, Double, SparseVector[Double])])
            (implicit randBasis: RandBasis = Rand)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    assert(vocabSize >= dimK * 2)

    val para_main: Double = (alpha0 + 1.0) / numDocs.toDouble
    val para_shift: Double = alpha0

    //val SEED_random: Long = System.currentTimeMillis
    val gaussianRandomMatrix: DenseMatrix[Double] = AlgebraUtil.gaussian(vocabSize, dimK * 2)
    val gaussianRandomMatrix_broadcasted: Broadcast[breeze.linalg.DenseMatrix[Double]] = sc.broadcast(gaussianRandomMatrix)
    val firstOrderMoments_broadcasted: Broadcast[breeze.linalg.DenseVector[Double]] = sc.broadcast(firstOrderMoments.toDenseVector)

    val M2_a_S: DenseMatrix[Double] = documents map {
      this_document => accumulate_M_mul_S(
        vocabSize, dimK * 2,
        alpha0,
        firstOrderMoments_broadcasted.value,
        gaussianRandomMatrix_broadcasted.value,
        this_document._3, this_document._2)
    } reduce(_ + _)

    M2_a_S :*= para_main
    val shiftedMatrix: breeze.linalg.DenseMatrix[Double] = firstOrderMoments * (firstOrderMoments.t * gaussianRandomMatrix)
    M2_a_S -= shiftedMatrix :* para_shift

    val QR(q: DenseMatrix[Double], _) = qr.reduced(M2_a_S)

    val M2_a_Q: DenseMatrix[Double] = documents map {
      this_document => accumulate_M_mul_S(
        vocabSize,
        dimK * 2, alpha0,
        firstOrderMoments_broadcasted.value,
        q,
        this_document._3, this_document._2)
    } reduce(_ + _)
    M2_a_Q :*= para_main
    val shiftedMatrix2: breeze.linalg.DenseMatrix[Double] = firstOrderMoments * (firstOrderMoments.t * q)
    M2_a_Q -= shiftedMatrix2 :* para_shift

    // Note: eigenvectors * Diag(eigenvalues) = M2_a_Q
    val EigSym(s: DenseVector[Double], u: DenseMatrix[Double]) = eigSym(q.t * M2_a_Q)
    val idx = argtopk(s, dimK)
    (q * u(::, idx).copy, s(idx).copy.toDenseVector)
  }



  private def accumulate_M_mul_S(dimVocab: Int, dimK: Int, alpha0: Double,
                                 m1: breeze.linalg.DenseVector[Double], S: breeze.linalg.DenseMatrix[Double], Wc: breeze.linalg.SparseVector[Double], len: Double) = {
    assert(dimVocab == Wc.length)
    assert(dimVocab == m1.length)
    assert(dimVocab == S.rows)
    assert(dimK == S.cols)
    val len_calibrated: Double = math.max(len, 3.0)

    val M2_a = breeze.linalg.DenseMatrix.zeros[Double](dimVocab, dimK)

    val norm_length: Double = 1.0 / (len_calibrated * (len_calibrated - 1.0))
    val data_mul_S: DenseVector[Double] = breeze.linalg.DenseVector.zeros[Double](dimK)

    var offset = 0
    while (offset < Wc.activeSize) {
      val token: Int = Wc.indexAt(offset)
      val count: Double = Wc.valueAt(offset)
      data_mul_S += S(token, ::).t.map(x => x * count)
      offset += 1
    }

    offset = 0
    while (offset < Wc.activeSize) {
      val token: Int = Wc.indexAt(offset)
      val count: Double = Wc.valueAt(offset)
      M2_a(token, ::) += (data_mul_S - S(token, ::).t).map(x => x * count * norm_length).t

      offset += 1
    }
    M2_a
  }


}
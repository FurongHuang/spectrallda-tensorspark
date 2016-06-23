package edu.uci.eecs.spectralLDA.utils

import breeze.linalg.eigSym.EigSym
import breeze.linalg.qr.QR
import breeze.linalg.{CSCMatrix, DenseMatrix, DenseVector, SparseVector, argtopk, eigSym, qr, svd}
import breeze.stats.distributions.{Rand, RandBasis}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD


object RandNLA {
  /*def whiten(sc: SparkContext,
                     alpha0: Double,
                     vocabSize: Int, dimK: Int,
                     numDocs: Long,
                     firstOrderMoments: DenseVector[Double],
                     documents: RDD[(Long, Double, SparseVector[Double])])
            (implicit randBasis: RandBasis = Rand)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    val para_main: Double = (alpha0 + 1.0) / numDocs.toDouble
    val para_shift: Double = alpha0

    val gaussianRandomMatrix: DenseMatrix[Double] = AlgebraUtil.gaussian(vocabSize, dimK * 2)
    val gaussianRandomMatrix_broadcasted: Broadcast[breeze.linalg.DenseMatrix[Double]] = sc.broadcast(gaussianRandomMatrix)

    val M2_a_S_1: CSCMatrix[Double] = documents map {
      this_document => accumulate_M_mul_S(
        vocabSize, dimK * 2,
        alpha0,
        gaussianRandomMatrix_broadcasted.value,
        this_document._3, this_document._2)
    } reduce(_ + _)

    val M2_a_S: DenseMatrix[Double] = M2_a_S_1.toDense * para_main
              - (firstOrderMoments * (firstOrderMoments.t * gaussianRandomMatrix)) * para_shift
    gaussianRandomMatrix_broadcasted.destroy

    val Q = AlgebraUtil.orthogonalizeMatCols(M2_a_S)

    val Q_broadcasted = sc.broadcast(Q)
    val M2_a_Q_1: CSCMatrix[Double] = documents map {
      this_document => accumulate_M_mul_S(
        vocabSize,
        dimK * 2, alpha0,
        Q_broadcasted.value,
        this_document._3, this_document._2)
    } reduce(_ + _)

    val M2_a_Q: DenseMatrix[Double] = M2_a_Q_1.toDense * para_main
          - (firstOrderMoments * (firstOrderMoments.t * Q)) * para_shift
    Q_broadcasted.destroy

    // Note: eigenvectors * Diag(eigenvalues) = M2_a_Q
    val svd.SVD(u: breeze.linalg.DenseMatrix[Double], s: breeze.linalg.DenseVector[Double], v: breeze.linalg.DenseMatrix[Double]) = svd(M2_a_Q.t * M2_a_Q)
    val eigenVectors: DenseMatrix[Double] = (M2_a_Q * u) * breeze.linalg.diag(s.map(entry => 1.0 / math.sqrt(entry)))
    val eigenValues: DenseVector[Double] = s.map(entry => math.sqrt(entry))
    (eigenVectors(::, 0 until dimK), eigenValues(0 until dimK))
  }*/

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

    val gaussianRandomMatrix: DenseMatrix[Double] = AlgebraUtil.gaussian(vocabSize, dimK * 2)
    val gaussianRandomMatrix_broadcasted: Broadcast[breeze.linalg.DenseMatrix[Double]] = sc.broadcast(gaussianRandomMatrix)

    val M2_a_S_1_rdd: RDD[(Int, DenseVector[Double])] = documents flatMap {
      this_document => accumulate_M_mul_S(
        vocabSize, dimK * 2,
        alpha0,
        gaussianRandomMatrix_broadcasted.value,
        this_document._3, this_document._2)
    } reduceByKey(_ + _)
    val M2_a_S_1: DenseMatrix[Double] = DenseMatrix.zeros[Double](vocabSize, dimK * 2)
    M2_a_S_1_rdd.collect.foreach {
      case (token, v) => M2_a_S_1(token, ::) := v.t
    }

    val M2_a_S: DenseMatrix[Double] = M2_a_S_1 * para_main
          - (firstOrderMoments * (firstOrderMoments.t * gaussianRandomMatrix)) * para_shift
    gaussianRandomMatrix_broadcasted.destroy

    val QR(q: DenseMatrix[Double], _) = qr.reduced(M2_a_S)
    val q_broadcasted = sc.broadcast(q)
    
    val M2_a_Q_1_rdd: RDD[(Int, DenseVector[Double])] = documents flatMap {
      this_document => accumulate_M_mul_S(
        vocabSize,
        dimK * 2, alpha0,
        q_broadcasted.value,
        this_document._3, this_document._2)
    } reduceByKey(_ + _)
    val M2_a_Q_1: DenseMatrix[Double] = DenseMatrix.zeros[Double](vocabSize, dimK * 2)
    M2_a_Q_1_rdd.collect.foreach {
      case (token, v) => M2_a_Q_1(token, ::) := v.t
    }
    val M2_a_Q = M2_a_Q_1 * para_main
          - (firstOrderMoments * (firstOrderMoments.t * q)) * para_shift

    // Note: eigenvectors * Diag(eigenvalues) = M2_a_Q
    val w = q.t * M2_a_Q
    val EigSym(s: DenseVector[Double], u: DenseMatrix[Double]) = eigSym((w + w.t) / 2.0)
    val idx = argtopk(s, dimK)

    val u_M2: DenseMatrix[Double] = q * u(::, idx).copy
    val s_M2: DenseVector[Double] = s(idx).copy.toDenseVector
    
    q_broadcasted.destroy
    (u_M2, s_M2)
  }



  private def accumulate_M_mul_S(dimVocab: Int, dimK: Int, alpha0: Double,
                                 S: breeze.linalg.DenseMatrix[Double],
                                 Wc: breeze.linalg.SparseVector[Double], len: Double)
        : Seq[(Int, DenseVector[Double])] = {
    assert(dimVocab == Wc.length)
    assert(dimVocab == S.rows)
    assert(dimK == S.cols)
    val len_calibrated: Double = math.max(len, 3.0)

    //val M2_a: CSCMatrix[Double] = CSCMatrix.zeros[Double](dimVocab, dimK)

    val norm_length: Double = 1.0 / (len_calibrated * (len_calibrated - 1.0))
    /*val data_mul_S: DenseVector[Double] = breeze.linalg.DenseVector.zeros[Double](dimK)

    var offset = 0
    while (offset < Wc.activeSize) {
      val token: Int = Wc.indexAt(offset)
      val count: Double = Wc.valueAt(offset)
      data_mul_S += S(token, ::).t.map(x => x * count)
      offset += 1
    }*/
    val data_mul_S: DenseVector[Double] = S.t * Wc

    /*offset = 0
    while (offset < Wc.activeSize) {
      val token: Int = Wc.indexAt(offset)
      val count: Double = Wc.valueAt(offset)
      val v: DenseVector[Double] = (data_mul_S - S(token, ::).t) * count * norm_length
      for (j <- 0 until v.length) {
        M2_a(token, j) = v(j)
      }

      offset += 1
    }
    M2_a*/

    Wc.activeIterator.toSeq.map { case (token, count) =>
      (token, (data_mul_S - S(token, ::).t) * count * norm_length)
    }
  }
}

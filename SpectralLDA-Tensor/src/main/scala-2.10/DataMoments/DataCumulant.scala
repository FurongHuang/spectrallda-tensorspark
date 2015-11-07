package DataMoments

/**
 * Data Cumulants Calculation.
 * Created by Furong Huang on 11/2/15.
 */

import Accumulator.{DenseMatrixAccumulatorParam, DenseVectorAccumulatorParam}
import Utils.AlgebraUtil
import breeze.linalg._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulator, SparkContext}
import scala.collection.mutable

class DataCumulant(sc: SparkContext, slices: Int, dimK: Int, alpha0: Double, tolerance: Double, documents:RDD[(Long, Double, SparseVector[Double])],dimVocab: Int,numDocs: Long ) extends Serializable {

  private val M1: Accumulator[DenseVector[Double]] = sc.accumulator(breeze.linalg.DenseVector.zeros[Double](dimVocab))(DenseVectorAccumulatorParam)
  println("Start calculating first order moments...")
  documents.foreach { case (_, length, vec) => M1 += update_firstOrderMoments(dimVocab, vec.toDenseVector, length) }
  private val firstOrderMoments: DenseVector[Double] = M1.value.map(x => x / numDocs.toDouble)
  println("Finished calculating first order moments.")

  val (thirdOrderMoments: DenseMatrix[Double], unwhiteningMatrix: DenseMatrix[Double]) = {
    println("Start calculating second order moments...")
    val (eigenVectors: DenseMatrix[Double], eigenValues: DenseVector[Double]) = whiten(sc, alpha0, dimVocab, dimK, numDocs, firstOrderMoments, documents)
    println("Finished calculating second order moments and whitening matrix.")

    println("Start whitening data with dimensionality reduction...")
    val whitenedData: RDD[(breeze.linalg.DenseVector[Double], Double)] = documents.map {
      case (_, length, vec) => (project(dimVocab, dimK, alpha0, eigenValues, eigenVectors, vec), length)
    }
    val firstOrderMoments_whitened: DenseVector[Double] = whitenedData.map(x => x._1 / x._2).reduce((a, b) => a :+ b).map(x => x / numDocs.toDouble)
    println("Finished whitening data.")

    println("Start calculating third order moments...")
    var Ta: Accumulator[DenseMatrix[Double]] = sc.accumulator(breeze.linalg.DenseMatrix.zeros[Double](dimK, dimK * dimK))(DenseMatrixAccumulatorParam)
    whitenedData.foreach{ case (vec, len) => Ta += update_thirdOrderMoments(dimK, alpha0, firstOrderMoments_whitened, vec, len)}

    val alpha0sq: Double = alpha0 * alpha0
    val Ta_shift = DenseMatrix.zeros[Double](dimK, dimK * dimK)
    for (id_i: Int <- 0 until dimK) {
      for (id_j: Int <- 0 until dimK) {
        for (id_l: Int <- 0 until dimK) {
          Ta_shift(id_i, id_j * dimK + id_l) += alpha0sq * firstOrderMoments_whitened(id_i) * firstOrderMoments_whitened(id_j) * firstOrderMoments_whitened(id_l)
        }
      }
    }
    println("Finished calculating third order moments.")
    val unwhiteningMatrix: breeze.linalg.DenseMatrix[Double] = eigenVectors * breeze.linalg.diag(eigenValues.map(x => scala.math.sqrt(x)))
    (Ta.value.map(x => x / numDocs.toDouble) - Ta_shift, unwhiteningMatrix)
  }

  private def whiten(sc: SparkContext, alpha0: Double, vocabSize: Int, dimK: Int, numDocs: Long, firstOrderMoments: breeze.linalg.DenseVector[Double], documents: RDD[(Long, Double, breeze.linalg.SparseVector[Double])]): (breeze.linalg.DenseMatrix[Double], breeze.linalg.DenseVector[Double]) = {
    val para_main: Double = (alpha0 + 1.0) / numDocs.toDouble
    val para_shift: Double = alpha0

    val SEED_random: Long = System.currentTimeMillis
    val gaussianRandomMatrix: DenseMatrix[Double] = AlgebraUtil.gaussian(vocabSize, dimK * 2, SEED_random)
    val gaussianRandomMatrix_broadcasted: Broadcast[breeze.linalg.DenseMatrix[Double]] = sc.broadcast(gaussianRandomMatrix)

    val firstOrderMoments_broadcasted: Broadcast[breeze.linalg.DenseVector[Double]] = sc.broadcast(firstOrderMoments.toDenseVector)
    // val documents_broadcasted: Broadcast[Array[(Long, Double, breeze.linalg.SparseVector[Double])]] = sc.broadcast(documents.collect())
    val M2_a_S: Accumulator[breeze.linalg.DenseMatrix[Double]] = sc.accumulator(breeze.linalg.DenseMatrix.zeros[Double](vocabSize, dimK * 2), "Second Order Moment multiplied with S: M2_a * S")(DenseMatrixAccumulatorParam)
    documents.foreach(this_document => M2_a_S += accumulate_M_mul_S(vocabSize, dimK * 2, alpha0, firstOrderMoments_broadcasted.value, gaussianRandomMatrix_broadcasted.value, this_document._3, this_document._2))

    M2_a_S.value *= para_main
    val shiftedMatrix: breeze.linalg.DenseMatrix[Double] = firstOrderMoments * (firstOrderMoments.t * gaussianRandomMatrix)
    M2_a_S.value -= shiftedMatrix :* para_shift

    val Q = AlgebraUtil.orthogonalizeMatCols(M2_a_S.value)
    val M2_a_Q: Accumulator[DenseMatrix[Double]] = sc.accumulator(breeze.linalg.DenseMatrix.zeros[Double](vocabSize, dimK * 2), "Second Order Moment multiplied with S: M2_a * S")(DenseMatrixAccumulatorParam)

    documents.foreach(this_document => M2_a_Q += accumulate_M_mul_S(vocabSize, dimK * 2, alpha0, firstOrderMoments, Q, this_document._3, this_document._2))
    M2_a_Q.value *= para_main
    val shiftedMatrix2: breeze.linalg.DenseMatrix[Double] = firstOrderMoments * (firstOrderMoments.t * Q)
    M2_a_Q.value -= shiftedMatrix2 :* para_shift

    // Note: eigenvectors * Diag(eigenvalues) = M2_a_Q
    val svd.SVD(u: breeze.linalg.DenseMatrix[Double], s: breeze.linalg.DenseVector[Double], v: breeze.linalg.DenseMatrix[Double]) = svd((M2_a_Q.value.t * M2_a_Q.value))
    val eigenVectors: DenseMatrix[Double] = (M2_a_Q.value * u) * breeze.linalg.diag(s.map(entry => 1.0 / math.sqrt(entry)))
    val eigenValues: DenseVector[Double] = s.map(entry => math.sqrt(entry))
    (eigenVectors(::, 0 until dimK), eigenValues(0 until dimK))
  }

  private def update_firstOrderMoments(dim: Int, Wc: breeze.linalg.DenseVector[Double], len: Double) = {
    val M1: DenseVector[Double] = Wc.map(x => x/len)
    M1
  }


  private def update_thirdOrderMoments(dimK: Int, alpha0: Double, m1: DenseVector[Double], Wc: DenseVector[Double], len: Double): DenseMatrix[Double] = {
    val len_calibrated: Double = math.max(len, 3.0)

    val scale3fac: Double = (alpha0 + 1.0) * (alpha0 + 2.0) / (2.0 * len_calibrated * (len_calibrated - 1.0) * (len_calibrated - 2.0))
    val scale2fac: Double = alpha0 * (alpha0 + 1.0) / (2.0 * len_calibrated * (len_calibrated - 1.0))
    val Ta = breeze.linalg.DenseMatrix.zeros[Double](dimK, dimK * dimK)

    for (i: Int <- 0 until dimK) {
      for (j: Int <- 0 until dimK) {
        for (l: Int <- 0 until dimK) {
          Ta(i, dimK * j + l) += scale3fac * Wc(i) * Wc(j) * Wc(l)

          Ta(i, dimK * j + l) -= scale2fac * Wc(i) * Wc(j) * m1(l)
          Ta(i, dimK * j + l) -= scale2fac * Wc(i) * m1(j) * Wc(l)
          Ta(i, dimK * j + l) -= scale2fac * m1(i) * Wc(j) * Wc(l)
        }
        Ta(i, dimK * i + j) -= scale3fac * Wc(i) * Wc(j)
        Ta(i, dimK * j + i) -= scale3fac * Wc(i) * Wc(j)
        Ta(i, dimK * j + j) -= scale3fac * Wc(i) * Wc(j)

        Ta(i, dimK * i + j) += scale2fac * Wc(i) * m1(j)
        Ta(i, dimK * j + i) += scale2fac * Wc(i) * m1(j)
        Ta(i, dimK * j + j) += scale2fac * m1(i) * Wc(j)
      }
      Ta(i, dimK * i + i) += 2.0 * scale3fac * Wc(i)
    }
    Ta
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

  private def project(dimVocab: Int, dimK: Int, alpha0: Double,
              eigenValues: breeze.linalg.DenseVector[Double], eigenVectors: breeze.linalg.DenseMatrix[Double],
              Wc: breeze.linalg.SparseVector[Double]): breeze.linalg.DenseVector[Double] = {
    var offset = 0
    val result = breeze.linalg.DenseVector.zeros[Double](dimK)
    while (offset < Wc.activeSize) {
      val token: Int = Wc.indexAt(offset)
      val count: Double = Wc.valueAt(offset)
      // val S_row = S(token,::)

      result += eigenVectors(token, ::).t.map(x => x * count)

      offset += 1
    }
    val whitenedData = result :/ eigenValues.map(x => math.sqrt(x) + tolerance)
    whitenedData
  }
}

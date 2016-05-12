package edu.uci.eecs.spectralLDA.datamoments

/**
 * Data Cumulants Calculation.
 * Created by Furong Huang on 11/2/15.
 */

import edu.uci.eecs.spectralLDA.utils.AlgebraUtil
import breeze.linalg._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import scala.collection.mutable

import scalaxy.loops._
import scala.language.postfixOps

case class DataCumulant(thirdOrderMoments: DenseMatrix[Double], unwhiteningMatrix: DenseMatrix[Double])
  extends Serializable


object DataCumulant {
  def getDataCumulant(dimK: Int,
                      alpha0: Double,
                      tolerance: Double,
                      documents: RDD[(Long, Double, SparseVector[Double])])
        : DataCumulant = {
    val sc: SparkContext = documents.sparkContext
    val dimVocab = documents.take(1)(0)._3.length
    val numDocs = documents.count()

    println("Start calculating first order moments...")
    val M1: DenseVector[Double] = documents map {
      case (_, length, vec) => update_firstOrderMoments(dimVocab, vec.toDenseVector, length)
    } reduce (_ + _)

    val firstOrderMoments: DenseVector[Double] = M1 / numDocs.toDouble
    println("Finished calculating first order moments.")

    val (thirdOrderMoments: DenseMatrix[Double], unwhiteningMatrix: DenseMatrix[Double]) = computeThirdOrderMoments(
      sc, alpha0, dimVocab, dimK,
      numDocs, firstOrderMoments, documents,
      tolerance
    )

    new DataCumulant(thirdOrderMoments, unwhiteningMatrix)
  }

  private def computeThirdOrderMoments(sc: SparkContext,
                                       alpha0: Double,
                                       dimVocab: Int, dimK: Int,
                                       numDocs: Long,
                                       firstOrderMoments: DenseVector[Double],
                                       documents: RDD[(Long, Double, SparseVector[Double])],
                                       tolerance: Double)
  : (DenseMatrix[Double], DenseMatrix[Double]) = {
    println("Start calculating second order moments...")
    val (eigenVectors: DenseMatrix[Double], eigenValues: DenseVector[Double]) = whiten(sc, alpha0,
      dimVocab, dimK, numDocs, firstOrderMoments, documents)
    println("Finished calculating second order moments and whitening matrix.")

    println("Start whitening data with dimensionality reduction...")
    val whitenedData: RDD[(DenseVector[Double], Double)] = documents.map {
      case (_, length, vec) => (project(dimVocab, dimK, alpha0, eigenValues, eigenVectors, vec)(tolerance), length)
    }
    val firstOrderMoments_whitened: DenseVector[Double] = whitenedData
      .map(x => x._1 / x._2)
      .reduce((a, b) => a :+ b)
      .map(x => x / numDocs.toDouble)
    println("Finished whitening data.")

    println("Start calculating third order moments...")
    val Ta: DenseMatrix[Double] = whitenedData map {
      case (vec, len) => update_thirdOrderMoments(
        dimK, alpha0,
        firstOrderMoments_whitened,
        vec, len)
    } reduce(_ + _)

    val alpha0sq: Double = alpha0 * alpha0
    val Ta_shift = DenseMatrix.zeros[Double](dimK, dimK * dimK)
    for (id_i <- 0 until dimK optimized) {
      for (id_j <- 0 until dimK optimized) {
        for (id_l <- 0 until dimK optimized) {
          Ta_shift(id_i, id_j * dimK + id_l) += (alpha0sq * firstOrderMoments_whitened(id_i)
              * firstOrderMoments_whitened(id_j) * firstOrderMoments_whitened(id_l))
        }
      }
    }
    println("Finished calculating third order moments.")
    val unwhiteningMatrix: breeze.linalg.DenseMatrix[Double] = eigenVectors * breeze.linalg.diag(eigenValues.map(x => scala.math.sqrt(x)))
    (Ta / numDocs.toDouble + Ta_shift, unwhiteningMatrix)
  }

  private def whiten(sc: SparkContext,
                     alpha0: Double,
                     vocabSize: Int, dimK: Int,
                     numDocs: Long,
                     firstOrderMoments: DenseVector[Double],
                     documents: RDD[(Long, Double, SparseVector[Double])])
  : (DenseMatrix[Double], DenseVector[Double]) = {
    val para_main: Double = (alpha0 + 1.0) / numDocs.toDouble
    val para_shift: Double = alpha0

    val SEED_random: Long = System.currentTimeMillis
    val gaussianRandomMatrix: DenseMatrix[Double] = AlgebraUtil.gaussian(vocabSize, dimK * 2, SEED_random)
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

  private def update_firstOrderMoments(dim: Int, Wc: breeze.linalg.DenseVector[Double], len: Double) = {
    val M1: DenseVector[Double] = Wc / len
    M1
  }


  private def update_thirdOrderMoments(dimK: Int, alpha0: Double, m1: DenseVector[Double], Wc: DenseVector[Double], len: Double): DenseMatrix[Double] = {
    val len_calibrated: Double = math.max(len, 3.0)

    val scale3fac: Double = (alpha0 + 1.0) * (alpha0 + 2.0) / (2.0 * len_calibrated * (len_calibrated - 1.0) * (len_calibrated - 2.0))
    val scale2fac: Double = alpha0 * (alpha0 + 1.0) / (2.0 * len_calibrated * (len_calibrated - 1.0))
    val Ta = breeze.linalg.DenseMatrix.zeros[Double](dimK, dimK * dimK)

    import scalaxy.loops._
    import scala.language.postfixOps
    for (i <- 0 until dimK optimized) {
      for (j <- 0 until dimK optimized) {
        for (l <- 0 until dimK optimized) {
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
                      eigenValues: breeze.linalg.DenseVector[Double],
                      eigenVectors: breeze.linalg.DenseMatrix[Double],
                      Wc: breeze.linalg.SparseVector[Double])
                     (implicit tolerance: Double)
  : breeze.linalg.DenseVector[Double] = {
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

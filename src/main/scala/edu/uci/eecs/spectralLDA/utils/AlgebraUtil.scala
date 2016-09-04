package edu.uci.eecs.spectralLDA.utils

/**
 * Utils for algebric operations.
 * Created by furongh on 11/2/15.
 */

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.{Gaussian, Rand, RandBasis}

import scalaxy.loops._
import scala.language.postfixOps

object AlgebraUtil {

  def to_invert(c: DenseMatrix[Double], b: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ctc: DenseMatrix[Double] = c.t * c
    val btb: DenseMatrix[Double] = b.t * b
    val to_be_inverted: DenseMatrix[Double] = ctc :* btb
    breeze.linalg.pinv(to_be_inverted)
  }

  def matrixNormalization(B: DenseMatrix[Double]): DenseMatrix[Double] = {
    val A: DenseMatrix[Double] = B.copy
    val colNorms: DenseVector[Double] = norm(A(::, *)).toDenseVector

    for (i <- 0 until A.cols optimized) {
      A(::, i) :/= colNorms(i)
    }
    A
  }

  def KhatrioRao(A: DenseVector[Double], B: DenseVector[Double]): DenseVector[Double] = {
    val Out: DenseMatrix[Double] = B * A.t
    Out.flatten()
  }

  def KhatrioRao(A: DenseMatrix[Double], B: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(B.cols == A.cols)
    val Out = DenseMatrix.zeros[Double](B.rows * A.rows, A.cols)
    for (i <- 0 until A.cols optimized) {
      Out(::, i) := KhatrioRao(A(::, i), B(::, i))

    }
    Out
  }


  def Multip_KhatrioRao(T: DenseVector[Double], C: DenseVector[Double], B: DenseVector[Double]): Double = {
    val longvec: DenseVector[Double] = KhatrioRao(C, B)
    T dot longvec
  }

  def Multip_KhatrioRao(T: DenseVector[Double], C: DenseMatrix[Double], B: DenseMatrix[Double]): DenseVector[Double] = {
    assert(B.cols == C.cols)
    val Out = DenseVector.zeros[Double](C.cols)
    for (i <- 0 until C.cols optimized) {
      Out(i) = Multip_KhatrioRao(T, C(::, i), B(::, i))

    }
    Out
  }


  def Multip_KhatrioRao(T: DenseMatrix[Double], C: DenseMatrix[Double], B: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(B.cols == C.cols)
    val Out = DenseMatrix.zeros[Double](C.cols, T.rows)
    for (i <- 0 until T.rows optimized) {
      val thisRowOfT: DenseVector[Double] = T(i, ::).t
      Out(::, i) := Multip_KhatrioRao(thisRowOfT, C, B)
    }
    Out.t
  }

  def isConverged(oldA: DenseMatrix[Double], newA: DenseMatrix[Double])
                 (implicit dotThreshold: Double = 0.99): Boolean = {
    if (oldA == null || oldA.size == 0) {
      return false
    }

    val dprod = diag(oldA.t * newA)
    println(s"dot(oldA, newA): ${diag(oldA.t * newA)}")

    all(dprod :> dotThreshold)
  }

  def Cumsum(xs: Array[Double]): Array[Double] = {
    xs.scanLeft(0.0)(_ + _).tail
  }

}

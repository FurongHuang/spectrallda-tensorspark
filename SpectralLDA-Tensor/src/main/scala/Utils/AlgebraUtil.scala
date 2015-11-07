package Utils

/**
 * Utils for algebric operations.
 * Created by furongh on 11/2/15.
 */

import breeze.linalg.{DenseMatrix, DenseVector}

object AlgebraUtil{
  private val SEED_random: Long = System.currentTimeMillis
  private val TOLERANCE: Double = 1.0e-9

  def gaussian(rows: Int, cols: Int, seed: Long = SEED_random): DenseMatrix[Double] = {
    val vectorizedOutputMatrix: Array[Double] = new Array[Double](rows * cols)
    val rand: scala.util.Random = new scala.util.Random(seed)
    for (i: Int <- vectorizedOutputMatrix.indices) {
      vectorizedOutputMatrix(i) = rand.nextGaussian()
    }
    val gaussianMatrix: DenseMatrix[Double] = new DenseMatrix(rows, cols, vectorizedOutputMatrix)
    matrixNormalization(gaussianMatrix)
  }

  def to_invert(c: DenseMatrix[Double], b: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ctc: DenseMatrix[Double] = c.t * c
    val btb: DenseMatrix[Double] = b.t * b
    val to_be_inverted: DenseMatrix[Double] = ctc :* btb
    return breeze.linalg.pinv(to_be_inverted)
  }


  def orthogonalizeMatCols(B: DenseMatrix[Double]): DenseMatrix[Double] = {
    val A:DenseMatrix[Double] = B.copy

    for (j: Int <- 0 until A.cols) {
      for (i: Int <- 0 until j) {
        val dotij = A(::, j) dot A(::, i)
        A(::, j) :-= (A(::, i) :* dotij)

      }
      val normsq_sqrt: Double = Math.sqrt(A(::, j) dot A(::, j))
      var scale: Double = if (normsq_sqrt > TOLERANCE) 1.0 / normsq_sqrt else 1e-12
      A(::, j) :*= scale
    }
    return A
  }


  def colWiseNorm2(A: breeze.linalg.DenseMatrix[Double]): breeze.linalg.DenseVector[Double] = {
    val normVec = breeze.linalg.DenseVector.zeros[Double](A.cols)
    for (i: Int <- 0 until A.cols) {
      val thisnorm: Double = Math.sqrt(A(::, i) dot A(::, i))
      normVec(i) = (if (thisnorm > TOLERANCE) thisnorm else TOLERANCE)
    }
    normVec
  }


  def matrixNormalization(B: DenseMatrix[Double]): DenseMatrix[Double] = {
    val A: DenseMatrix[Double] = B.copy
    for (i: Int <- 0 until A.cols) {
      val thisnorm: Double = Math.sqrt(A(::, i) dot A(::, i))
      A(::, i) :*= (if (thisnorm > TOLERANCE) (1.0 / thisnorm) else TOLERANCE)
    }
    return A
  }

  def KhatrioRao(A: DenseVector[Double], B: DenseVector[Double]): DenseVector[Double] = {
    val Out: DenseMatrix[Double] = B * A.t
    Out.flatten()
  }

  def KhatrioRao(A: DenseMatrix[Double], B: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(B.cols == A.cols)
    val Out = DenseMatrix.zeros[Double](B.rows * A.rows, A.cols)
    for (i: Int <- 0 until A.cols) {
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
    for (i: Int <- 0 until C.cols) {
      Out(i) = Multip_KhatrioRao(T, C(::, i), B(::, i))

    }
    Out
  }


  def Multip_KhatrioRao(T: DenseMatrix[Double], C: DenseMatrix[Double], B: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(B.cols == C.cols)
    val Out = DenseMatrix.zeros[Double](C.cols, T.rows)
    for (i: Int <- 0 until T.rows) {
      val thisRowOfT: DenseVector[Double] = T(i, ::).t
      Out(::, i) := Multip_KhatrioRao(thisRowOfT, C, B)
    }
    Out.t
  }

  def isConverged(oldA: DenseMatrix[Double], newA: DenseMatrix[Double]): Boolean = {
    if (oldA == null || oldA.size == 0) {
      return false
    }
    val numerator: Double = breeze.linalg.norm(newA.toDenseVector - oldA.toDenseVector)
    val denominator: Double = breeze.linalg.norm(newA.toDenseVector)
    val delta: Double = numerator / denominator
    return delta < TOLERANCE
  }


  def Cumsum(xs: Array[Double]): Array[Double] = {
    // def apply(xs : Seq[Int]) : Seq[Int] =
    //   xs.scanLeft(0)(_ + _).tail
    xs.scanLeft(0.0)(_ + _).tail
  }

}

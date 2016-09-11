package edu.uci.eecs.spectralLDA.utils

import breeze.linalg.{*, CSCMatrix, Counter, DenseMatrix, DenseVector, SparseVector, Tensor, max, norm}
import breeze.math.{Complex, Semiring}
import breeze.storage.Zero

import scala.reflect.ClassTag
import scalaxy.loops._
import scala.language.postfixOps

object TensorOps {
  def matrixNorm(m: DenseMatrix[Complex]): Double = {
    norm(norm(m(::, *)).toDenseVector)
  }

  def dmatrixNorm(m: DenseMatrix[Double]): Double = {
    norm(norm(m(::, *)).toDenseVector)
  }

  def unfoldTensor3d[@specialized(Double) V : ClassTag : Zero : Numeric : Semiring]
        (t: Tensor[Seq[Int], V], n: Seq[Int]): DenseMatrix[V] = {
    assert(n.length == 3)
    val m = DenseMatrix.zeros[V](n(0), n(1) * n(2))
    for (i <- 0 until n(0); j <- 0 until n(1); k <- 0 until n(2)) {
      m(i, k * n(1) + j) = t(Seq(i, j, k))
    }
    m
  }

  def tensor3dFromUnfolded[@specialized(Double) V : ClassTag : Zero : Numeric : Semiring]
      (g: DenseMatrix[V], n: Seq[Int]): Tensor[Seq[Int], V] = {
    assert(n.length == 3)
    val t: Tensor[Seq[Int], V] = Counter[Seq[Int], V]
    for (i <- 0 until n(0); j <- 0 until n(1); k <- 0 until n(2)) {
      t(Seq(i, j, k)) = g(i, k * n(1) + j)
    }
    t
  }

  /** makes 3rd-order rank-1 tensor $$x\otimes y\otimes z$$, returns the unfolded version */
  def makeRankOneTensor3d[@specialized(Double) V : ClassTag : Zero : Numeric : Semiring]
      (x: DenseVector[V], y: DenseVector[V], z: DenseVector[V]): DenseMatrix[V] = {
    val d1 = x.length
    val d2 = y.length
    val d3 = z.length

    val result = DenseMatrix.zeros[V](d1, d2 * d3)

    val evV = implicitly[Numeric[V]]
    import evV._

    for (i <- 0 until d1 optimized) {
      for (j <- 0 until d2 optimized) {
        for (k <- 0 until d3 optimized) {
          result(i, k * d2 + j) = x(d1) * y(d2) * z(d3)
        }
      }
    }

    result
  }

  /** Khatri-Rao product */
  def krprod[@specialized(Double) V : ClassTag : Zero : Numeric : Semiring]
       (x: DenseVector[V], y: DenseVector[V]): DenseVector[V] = {
    val ev = implicitly[Numeric[V]]
    import ev._

    val seq = for (i <- 0 until x.size; j <- 0 until y.size) yield x(i) * y(j)
    DenseVector(seq: _*)
  }

  /** Khatri-Rao product */
  def krprod[@specialized(Double) V : ClassTag : Zero : Numeric : Semiring]
        (A: DenseMatrix[V], B: DenseMatrix[V]): DenseMatrix[V] = {
    assert(A.cols == B.cols)
    val result = DenseMatrix.zeros[V](A.rows * B.rows, A.cols)
    for (i <- 0 until A.cols) {
      result(::, i) := krprod[V](A(::, i), B(::, i))
    }
    result
  }

  def to_invert(c: DenseMatrix[Double], b: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ctc: DenseMatrix[Double] = c.t * c
    val btb: DenseMatrix[Double] = b.t * b
    val to_be_inverted: DenseMatrix[Double] = ctc :* btb
    breeze.linalg.pinv(to_be_inverted)
  }

  /** tensor product v * v.t given sparse vector v */
  def spVectorTensorProd2d(v: SparseVector[Double]): CSCMatrix[Double] = {
    val prod: CSCMatrix[Double] = CSCMatrix.zeros[Double](v.length, v.length)
    for (i <- 0 until v.activeSize; j <- 0 until v.activeSize) {
      prod(v.indexAt(i), v.indexAt(j)) = v.valueAt(i) * v.valueAt(j)
    }
    prod
  }
}
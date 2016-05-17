package edu.uci.eecs.spectralLDA.utils

import breeze.linalg.{*, Counter, DenseMatrix, DenseVector, Tensor, max, norm}
import breeze.math.{Complex, Semiring}
import breeze.storage.Zero

import scala.reflect.ClassTag


object TensorOps {
  def matrixNorm(m: DenseMatrix[Complex]): Double = {
    max(norm(m(::, *)))
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

  def makeRank1Tensor[@specialized(Double) V : ClassTag : Zero : Numeric : Semiring]
       (v: Seq[DenseVector[V]]): Tensor[Seq[Int], V] = {
    val n: Seq[Int] = v map { _.length }
    val t: Tensor[Seq[Int], V] = Counter[Seq[Int], V]

    val multiIndexRange = cartesianProduct(n map { Range(0, _) })
    for (i <- multiIndexRange) {
      val l = for {
        k <- n.indices
      } yield v(k)(i(k))

      t(i) = l.product
    }
    t
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

  private def cartesianProduct[A](xs: Traversable[Traversable[A]]): Seq[Seq[A]] = {
    xs.foldLeft(Seq(Seq.empty[A])){
      (x, y) => for (a <- x.view; b <- y) yield a :+ b
    }
  }
}
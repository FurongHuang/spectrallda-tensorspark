package edu.uci.eecs.spectralLDA.utils

import breeze.linalg.{*, Counter, DenseMatrix, Tensor, max, norm}
import breeze.math.Complex


object TensorOps {
  def matrixNorm(m: DenseMatrix[Complex]): Double = {
    max(norm(m(::, *)))
  }

  def unfoldTensor3d(t: Tensor[Seq[Int], Double], n: Seq[Int]): DenseMatrix[Double] = {
    assert(n.length == 3)
    val m = DenseMatrix.zeros[Double](n(0), n(1) * n(2))
    for (i <- 0 until n(0); j <- 0 until n(1); k <- 0 until n(2)) {
      m(i, j * n(2) + k) = t(Seq(i, j, k))
    }
    m
  }

  def tensor3dFromUnfolded(g: DenseMatrix[Double], n: Seq[Int]): Tensor[Seq[Int], Double] = {
    assert(n.length == 3)
    val t: Tensor[Seq[Int], Double] = Counter()
    for (i <- 0 until n(0); j <- 0 until n(1); k <- 0 until n(2)) {
      t(Seq(i, j, k)) = g(i, j * n(2) + k)
    }
    t
  }
}
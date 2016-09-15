package edu.uci.eecs.spectralLDA.utils

import breeze.linalg._
import breeze.math.Semiring
import breeze.storage.Zero
import org.scalatest._

import scala.reflect.ClassTag
import scalaxy.loops._
import scala.language.postfixOps


class TensorOpsTest extends FlatSpec with Matchers {
  def expectedRankOneTensor3d[@specialized(Double) V : ClassTag : Zero : Numeric : Semiring]
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
          result(i, k * d2 + j) = x(i) * y(j) * z(k)
        }
      }
    }

    result
  }

  "3rd-order vector outer product" should "be correct" in {
    val x = DenseVector.rand[Double](50)
    val y = DenseVector.rand[Double](50)
    val z = DenseVector.rand[Double](50)

    val tensor = TensorOps.makeRankOneTensor3d(x, y, z)
    val expected = expectedRankOneTensor3d(x, y, z)

    val diff = tensor - expected
    norm(norm(diff(::, *)).toDenseVector) should be <= 1e-8
  }
}
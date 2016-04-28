package edu.uci.eecs.spectralLDA.sketch

import breeze.linalg._
import breeze.stats.distributions._
import breeze.math._
import breeze.numerics._
import org.scalatest._
import org.scalatest.Matchers._


class TensorSketchTest extends FlatSpec with Matchers {
  "A 2x2x2 tensor's sketch" should "be correct" in {
    val n: Seq[Int] = Seq(2, 2, 2)
    val b = 4
    val B = 3

    val xi: Tensor[(Int, Int, Int),Double] = Counter(
      (0,2,1) -> -1.0, (1,1,1) -> 1.0, (2,2,1) -> 1.0,
      (1,1,0) -> 1.0, (2,0,0) -> 1.0, (0,1,0) -> 1.0,
      (0,0,0) -> -1.0, (0,1,1) -> 1.0, (1,2,1) -> -1.0,
      (2,0,1) -> -1.0, (1,2,0) -> -1.0, (0,0,1) -> 1.0,
      (2,2,0) -> 1.0, (2,1,1) -> 1.0, (0,2,0) -> -1.0,
      (1,0,0) -> -1.0, (1,0,1) -> -1.0, (2,1,0) -> 1.0
    )
    val h: Tensor[(Int, Int, Int),Int] = Counter(
      (0,2,1) -> 1, (1,1,1) -> 0, (2,2,1) -> 2,
      (1,1,0) -> 3, (2,0,0) -> 1, (0,1,0) -> 0,
      (0,0,0) -> 1, (0,1,1) -> 3, (1,2,1) -> 1,
      (2,0,1) -> 2, (1,2,0) -> 2, (0,0,1) -> 3,
      (2,2,0) -> 0, (2,1,1) -> 2, (0,2,0) -> 1,
      (1,0,0) -> 1, (1,0,1) -> 0, (2,1,0) -> 2
    )
    val sketch = new TensorSketch[Double, Double](n, b, B, xi, h)

    val a: Tensor[Seq[Int], Double] = Counter(
      Seq(0,0,0) -> 3, Seq(0,1,0) -> 5, Seq(1,0,0) -> 7,
      Seq(1,1,0) -> 9, Seq(0,0,1) -> 2, Seq(0,1,1) -> 4,
      Seq(1,0,1) -> 6, Seq(1,1,1) -> 8
    )

    val s = sketch.sketch(a)
    val recovered_a = sketch.recover(s)

    s should equal (new DenseMatrix[Double](3, 4,
      Array[Double](-13, 6, -16, 9, 17, 6,
        5, 16, -14, -17, 5, 8)
    ))
    recovered_a should equal (Counter(
      List(0, 1, 1) -> 9.0, List(0, 0, 0) -> 8.0, List(1, 1, 0) -> 16.0,
      List(1, 0, 0) -> 16.0, List(0, 1, 0) -> 8.0, List(1, 0, 1) -> 13.0,
      List(1, 1, 1) -> 17.0, List(0, 0, 1) -> 6.0
    ))
  }

  "Two 10x10x10 random tensors' sketches inner product" should "be close to the tensors' inner product" in {
    val n: Seq[Int] = Seq(10, 10, 10)
    val b = 32
    val B = 40

    val sketch = TensorSketch[Double, Double](n, b, B)

    // Two uniformly random tensors
    val t1: Tensor[Seq[Int], Double] = Counter()
    val t2: Tensor[Seq[Int], Double] = Counter()

    val uniform = new Uniform(0.0, 0.5)
    val u1 = uniform.sample(n(0))
    val u2 = uniform.sample(n(0))
    for (i <- 0 until n(0)) {
      for (j <- 0 until n(1)) {
        for (k <- 0 until n(2)) {
          t1(Seq(i, j, k)) = u1(i) * u1(j) * u1(k)
          if (i <= j && j <= k)
            t2(Seq(i, j, k)) = u2(i) * u2(j) * u2(k)
        }
      }
    }

    // Sketch and compare the inner products
    val s1 = sketch.sketch(t1)
    val s2 = sketch.sketch(t2)

    val innerProductTensors = sum(
      for {
        i <- 0 until n(0)
        j <- 0 until n(1)
        k <- 0 until n(2)
      } yield t1(Seq(i, j, k)) * t2(Seq(i, j, k))
    )
    val innerProductSketches = sum(s1 :* s2) / B

    val delta = innerProductSketches - innerProductTensors
    info("Difference of inner products: %f" format delta)
    abs(delta) should be <= 0.05
  }
}
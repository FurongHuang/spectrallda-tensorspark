package edu.uci.eecs.spectralLDA.sketch

import breeze.linalg._
import breeze.stats.distributions.Uniform
import breeze.signal.fourierTr
import breeze.math.Complex
import breeze.numerics.abs
import edu.uci.eecs.spectralLDA.utils.TensorOps
import org.scalatest._
import org.scalatest.Matchers._


class TensorSketchTest extends FlatSpec with Matchers {
  "Tensor's sketch" should "be correct" in {
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
    val sketcher = new TensorSketcher[Double, Double](n, b, B, xi, h)

    val a: Tensor[Seq[Int], Double] = Counter(
      Seq(0,0,0) -> 3, Seq(0,1,0) -> 5, Seq(1,0,0) -> 7,
      Seq(1,1,0) -> 9, Seq(0,0,1) -> 2, Seq(0,1,1) -> 4,
      Seq(1,0,1) -> 6, Seq(1,1,1) -> 8
    )

    val s = sketcher.sketch(a)
    val recovered_a = sketcher.recover(s)

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

  "The inner product of two 3d tensors' sketches" should
    "be close to the tensors' inner product" in {
    val n: Seq[Int] = Seq(10, 10, 10)
    val b = 32
    val B = 40

    val sketcher = TensorSketcher[Double, Double](n, b, B)

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
    val s1 = sketcher.sketch(t1)
    val s2 = sketcher.sketch(t2)

    val innerProductTensors = sum(
      for {
        i <- 0 until n(0)
        j <- 0 until n(1)
        k <- 0 until n(2)
      } yield t1(Seq(i, j, k)) * t2(Seq(i, j, k))
    )
    val innerProductSketches = sum(s1 :* s2) / B

    val delta = innerProductSketches - innerProductTensors
    //info("Difference of inner products: %f" format delta)
    abs(delta) should be <= 0.05
  }

  "Sketch of rank-1 3d tensor" should "be the convolution of the sketches along each dimension" in {
    val n = Seq(3, 3, 3)
    val sketcher = TensorSketcher[Double, Double](
      n = n,
      B = 10,
      b = Math.pow(2, 2).toInt
    )

    val v: Seq[DenseVector[Double]] = Seq(
      DenseVector[Double](0.2, 0.3, 0.5),
      DenseVector[Double](0.3, 0.5, 0.7),
      DenseVector[Double](0.5, 0.7, 0.9)
    )

    val t: Tensor[Seq[Int], Double] = Counter()
    for (i <- 0 until n(0); j <- 0 until n(1); k <- 0 until n(2)) {
      t(Seq(i, j, k)) = v(0)(i) * v(1)(j) * v(2)(k)
    }

    val sketch_v = (0 until 3).map { d => sketcher.sketch(v(d), d) }
    val fft_sketch_v = sketch_v.map { A: DenseMatrix[Double] => fourierTr(A(*, ::)) }
    val prod_fft_sketch_v: DenseMatrix[Complex] = fft_sketch_v(0) :* fft_sketch_v(1) :* fft_sketch_v(2)

    val sketch_t = sketcher.sketch(t)
    val fft_sketch_t: DenseMatrix[Complex] = fourierTr(sketch_t(*, ::))

    TensorOps.matrixNorm(fft_sketch_t - prod_fft_sketch_v) should be <= 1e-6
  }

  "Sketch of whitened rank-1 3d tensor" should
    "be the convolution of sketches of whitened vectors along each dimension" in {
    val v: Seq[DenseVector[Double]] = Seq(
      DenseVector.rand(50),
      DenseVector.rand(50),
      DenseVector.rand(50)
    )
    val orig_n: Seq[Int] = v map { _.length }

    val t: Tensor[Seq[Int], Double] = Counter()
    for (i <- 0 until orig_n(0); j <- 0 until orig_n(1); k <- 0 until orig_n(2)) {
      t(Seq(i, j, k)) = v(0)(i) * v(1)(j) * v(2)(k)
    }

    val W: DenseMatrix[Double] = DenseMatrix.rand[Double](50, 5)
    val n: Seq[Int] = Seq(5, 5, 5)

    val whitened_v = v map { W.t * _ }
    val whitened_t = TensorOps.tensor3dFromUnfolded(
      W.t * TensorOps.unfoldTensor3d(t, orig_n) * kron(W.t, W.t).t,
      n
    )

    val sketcher = TensorSketcher[Double, Double](
      n = Seq(5, 5, 5),
      B = 10,
      b = Math.pow(2, 4).toInt
    )

    val sketch_whitened_v = (0 until 3).map { d => sketcher.sketch(whitened_v(d), d) }
    val fft_sketch_whitened_v = sketch_whitened_v.map { A: DenseMatrix[Double] => fourierTr(A(*, ::)) }
    val prod_fft_sketch_whitened_v: DenseMatrix[Complex] = (fft_sketch_whitened_v(0)
      :* fft_sketch_whitened_v(1) :* fft_sketch_whitened_v(2))

    val sketch_whitened_t = sketcher.sketch(whitened_t)
    val fft_sketch_whitened_t: DenseMatrix[Complex] = fourierTr(sketch_whitened_t(*, ::))

    TensorOps.matrixNorm(fft_sketch_whitened_t - prod_fft_sketch_whitened_v) should be <= 1e-6
  }

  "Sketch of whitened diagonal matrix" should "be the expected sum of sketches" in {
    val v: DenseVector[Double] = DenseVector.rand(50)
    val t: DenseMatrix[Double] = diag(v)

    val sketcher = TensorSketcher[Double, Double](
      n = Seq(5, 5),
      B = 10,
      b = Math.pow(2, 4).toInt
    )

    val W = DenseMatrix.rand[Double](50, 5)

    val whitened_t = W.t * t * W

    val sketch_whitened_t = sketcher.sketch(whitened_t)
    val fft_sketch_whitened_t = fourierTr(sketch_whitened_t(*, ::))

    val sum_fft_sketches = fft_sketch_whitened_t * Complex(0, 0)
    for (k <- 0 until v.length) {
      val sketch_w_i = (0 until 2) map { sketcher.sketch(W(k, ::).t, _) }
      val fft_sketch_w_i: Seq[DenseMatrix[Complex]] = sketch_w_i
        .map { A: DenseMatrix[Double] => fourierTr(A(*, ::)) }

      sum_fft_sketches :+= (fft_sketch_w_i(0) :* fft_sketch_w_i(1)) * Complex(v(k), 0)
    }

    TensorOps.matrixNorm(fft_sketch_whitened_t - sum_fft_sketches) should be <= 1e-6
  }
}
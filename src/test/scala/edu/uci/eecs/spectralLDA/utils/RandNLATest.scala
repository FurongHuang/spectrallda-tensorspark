package edu.uci.eecs.spectralLDA.utils

import breeze.linalg._
import breeze.linalg.qr.QR
import breeze.stats.distributions.{Gaussian, RandBasis, ThreadLocalRandomGenerator, Uniform}
import edu.uci.eecs.spectralLDA.testharness.Context
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.SparkContext
import org.scalatest._


class RandNLATest extends FlatSpec with Matchers {

  private val sc: SparkContext = Context.getSparkContext

  "M2 sketching" should "be correct" in {
    val a1 = SparseVector(DenseVector.rand[Double](100).toArray)
    val a2 = SparseVector(DenseVector.rand[Double](100).toArray)
    val a3 = SparseVector(DenseVector.rand[Double](100).toArray)

    val docs = Seq((1000L, a1), (1001L, a2), (1002L, a3))
    val docsRDD = sc.parallelize(docs)

    // Random Gaussian matrix
    val g = DenseMatrix.rand[Double](100, 50, Gaussian(mu = 0.0, sigma = 1.0))

    val result = DenseMatrix.zeros[Double](100, 50)
    docsRDD
      .flatMap {
        case (id: Long, w: SparseVector[Double]) => RandNLA.accumulate_M_mul_S(g, w, sum(w))
      }
      .reduceByKey(_ + _)
      .collect
      .foreach {
        case (r: Int, a: DenseVector[Double]) => result(r, ::) := a.t
      }

    val m2 = docsRDD
      .map {
        case (id: Long, w: SparseVector[Double]) =>
          val l = sum(w)
          (w * w.t - diag(w)) / (l * (l - 1.0))
      }
      .reduce(_ + _)
    val expectedResult = m2 * g

    val diff: DenseMatrix[Double] = result - expectedResult
    val normDiff: Double = norm(norm(diff(::, *)).toDenseVector)
    normDiff should be <= 1e-8
  }

  "Randomised Power Iteration method" should "be approximately correct" in {
    implicit val randBasis: RandBasis =
      new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(234787)))

    val n = 100
    val k = 5

    val alpha: DenseVector[Double] = DenseVector[Double](25.0, 20.0, 15.0, 10.0, 5.0)
    val beta: DenseMatrix[Double] = DenseMatrix.rand(n, k, Uniform(0.0, 1.0))

    val norms = norm(beta(::, *)).toDenseVector
    for (j <- 0 until k) {
      beta(::, j) /= norms(j)
    }

    val a: DenseMatrix[Double] = beta * diag(alpha) * beta.t
    val sigma: DenseMatrix[Double] = DenseMatrix.rand(n, k, Gaussian(mu = 0.0, sigma = 1.0))
    val y = a * sigma
    val QR(q: DenseMatrix[Double], _) = qr.reduced(y)

    val (s: DenseVector[Double], u: DenseMatrix[Double]) = RandNLA.decomp2(a * q, q)

    val diff_a = u * diag(s) * u.t - a
    val norm_diff_a = norm(norm(diff_a(::, *)).toDenseVector)

    norm_diff_a should be <= 1e-8
  }
}


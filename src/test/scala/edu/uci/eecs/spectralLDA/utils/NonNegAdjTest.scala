package edu.uci.eecs.spectralLDA.utils

import org.scalatest._
import breeze.linalg._


class NonNegAdjTest extends FlatSpec with Matchers {
  "Simple non-negative adjustment to vectors" should "be correct" in {
    val inputVectors = Seq(
      DenseVector[Double](0.5, 0.5, 0.5),
      DenseVector[Double](0.5, 0.5, -0.5),
      DenseVector[Double](0.3, 0.3, -0.3, -0.3, 0.1, 0.1),
      DenseVector[Double](0.3, 0.3, 0.3, 0.3, -0.3, -0.3, -0.3, -0.3, 0.1, 0.1, 0.1, 0.1),
      DenseVector[Double](0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, -0.3, -0.3, -0.3, -0.3),
      DenseVector[Double](10, 10, 10, 10, 30, 30, 30, 30, -30, -30, -30, -30)
    )

    val expectedAdjustedVectors = Seq(
      DenseVector(0.33333333333333337, 0.33333333333333337, 0.33333333333333337),
      DenseVector(0.5, 0.5, 0.0),
      DenseVector(0.35, 0.35, 0.0, 0.0, 0.15000000000000002, 0.15000000000000002),
      DenseVector(0.22499999999999995, 0.22499999999999995, 0.22499999999999995, 0.22499999999999995,
        0.0, 0.0, 0.0, 0.0, 0.024999999999999967, 0.024999999999999967, 0.024999999999999967, 0.024999999999999967),
      DenseVector(0.024999999999999967, 0.024999999999999967, 0.024999999999999967, 0.024999999999999967,
        0.22499999999999995, 0.22499999999999995, 0.22499999999999995, 0.22499999999999995, 0.0, 0.0, 0.0, 0.0),
      DenseVector(0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0)
    )

    for ((v1, v2) <- inputVectors.zip(expectedAdjustedVectors)) {
      norm(NonNegativeAdjustment.simplexProj(v1)._1 - v2) should be <= 1e-8
    }
  }
}
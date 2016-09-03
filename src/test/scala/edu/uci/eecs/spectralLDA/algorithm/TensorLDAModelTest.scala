package edu.uci.eecs.spectralLDA.algorithm

import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, norm}
import breeze.numerics.abs
import org.scalatest._
import org.apache.spark.SparkContext
import edu.uci.eecs.spectralLDA.testharness.Context

class TensorLDAModelTest extends FlatSpec with Matchers {

  private val sc: SparkContext = Context.getSparkContext

  "Multinomial log-likelihood" should "be correct" in {
    val p = DenseVector[Double](0.2, 0.5, 0.3)
    val x1 = DenseVector[Double](20, 50, 30)
    val x2 = DenseVector[Double](40, 40, 20)

    abs(TensorLDAModel.multinomialLogLikelihood(p, x1) - (-4.697546)) should be <= 1e-6
    abs(TensorLDAModel.multinomialLogLikelihood(p, x2) - (-15.42038)) should be <= 1e-6
  }
}


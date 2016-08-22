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

  "Topic distribution EM inference" should "be loosely close to expected" in {
    val beta = new DenseMatrix[Double](5, 3, Array[Double]
      (0.6, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.6, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.6))
    val alpha = DenseVector[Double](20.0, 50.0, 30.0)

    val wordCounts1 = SparseVector[Double](70, 20, 10, 0, 0)
    val wordCounts2 = SparseVector[Double](10, 20, 70, 10, 20)

    val ldaModel = new TensorLDAModel(beta, alpha,
      DenseMatrix.eye[Double](5), DenseVector.ones[Double](5))(smoothing = 0.0)

    val topicDistribution1 = ldaModel.inferTopicDistribution(ldaModel.smoothedBeta, wordCounts1, maxIterationsEM = 20)
    val topicDistribution2 = ldaModel.inferTopicDistribution(ldaModel.smoothedBeta, wordCounts2, maxIterationsEM = 20)

    val expectedTopicDistribution1 = DenseVector[Double](0.7, 0.2, 0.1)
    val expectedTopicDistribution2 = DenseVector[Double](0.1, 0.7, 0.2)

    norm(expectedTopicDistribution1 - topicDistribution1) should be <= 0.25
    norm(expectedTopicDistribution2 - topicDistribution2) should be <= 0.25
  }
}


package edu.uci.eecs.spectralLDA.algorithm

import breeze.linalg._
import breeze.stats.distributions.{Dirichlet, Multinomial, RandBasis, ThreadLocalRandomGenerator}
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher
import org.scalatest._
import org.scalatest.Matchers._
import org.apache.spark.SparkContext
import edu.uci.eecs.spectralLDA.testharness.Context
import org.apache.commons.math3.random.MersenneTwister

class TensorLDASketchTest extends FlatSpec with Matchers {

  private val sc: SparkContext = Context.getSparkContext

  def simulateLDAData(alpha: DenseVector[Double],
                      allTokenDistributions: DenseMatrix[Double],
                      numDocuments: Int,
                      numTokensPerDocument: Int)
  : Seq[(Long, SparseVector[Double])] = {
    assert(alpha.size == allTokenDistributions.cols)
    val k = alpha.size
    val V = allTokenDistributions.rows

    // Simulate the word histogram of each document
    val dirichlet = Dirichlet(alpha)
    val wordCounts: Seq[(Long, SparseVector[Double])] = for {
      d <- 0 until numDocuments

      topicDistribution: DenseVector[Double] = dirichlet.sample()
      tokenDistribution: DenseVector[Double] = allTokenDistributions * topicDistribution
      tokens = Multinomial(tokenDistribution) sample numTokensPerDocument

      c = SparseVector.zeros[Double](V)
      tokensCount = tokens foreach { t => c(t) += 1.0 }
    } yield (d.toLong, c)

    wordCounts
  }

  "Simulated LDA" should "be recovered" in {
    val alpha: DenseVector[Double] = DenseVector[Double](5.0, 10.0, 20.0)
    val allTokenDistributions: DenseMatrix[Double] = new DenseMatrix[Double](5, 3,
      Array[Double](0.6, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.6, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.6))

    val documents = simulateLDAData(
      alpha,
      allTokenDistributions,
      numDocuments = 2000,
      numTokensPerDocument = 1000
    )
    val documentsRDD = sc.parallelize(documents)

    implicit val randBasis: RandBasis =
      new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(347529)))

    val sketcher = TensorSketcher[Double, Double](
      n = Seq(3, 3, 3),
      B = 100,
      b = Math.pow(2, 8).toInt
    )

    val tensorLDA = new TensorLDASketch(
      dimK = 3,
      alpha0 = sum(alpha),
      sketcher = sketcher,
      maxIterations = 200,
      nonNegativeDocumentConcentration = false,
      randomisedSVD = false
    )

    val (fitted_beta: DenseMatrix[Double], fitted_alpha: DenseVector[Double]) = tensorLDA.fit(documentsRDD)

    // Rearrange the elements/columns of fitted_alpha and fitted_beta
    // to the order of initial alpha and beta
    val i = argsort(fitted_alpha)
    val sorted_beta = fitted_beta(::, i).toDenseMatrix
    // if one vector is all negative, multiply it by -1 to turn it positive
    for (j <- 0 until sorted_beta.cols) {
      if (max(sorted_beta(::, j)) <= 0.0) {
        sorted_beta(::, j) :*= -1.0
      }
    }
    val sorted_alpha = fitted_alpha(i).toDenseVector

    info(s"Expecting alpha: $alpha")
    info(s"Obtained alpha: $sorted_alpha")
    info(s"Expecting beta:\n$allTokenDistributions")
    info(s"Obtained beta:\n$sorted_beta")

    val diff_beta: DenseMatrix[Double] = sorted_beta - allTokenDistributions
    val diff_alpha: DenseVector[Double] = sorted_alpha - alpha

    val norm_diff_beta = norm(norm(diff_beta(::, *)).toDenseVector)
    val norm_diff_alpha = norm(diff_alpha)

    norm_diff_beta should be <= 0.2
    norm_diff_alpha should be <= 4.0
  }
}
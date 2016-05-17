package edu.uci.eecs.spectralLDA.algorithm

import breeze.linalg._
import breeze.stats.distributions.{Dirichlet, Multinomial}
import breeze.numerics.sqrt
import breeze.signal.fourierTr
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher
import org.scalatest._
import org.scalatest.Matchers._

import org.apache.spark.SparkContext
import edu.uci.eecs.spectralLDA.testharness.Context

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

  /*
  "Simulated LDA" should "be recovered" in {
    val alpha: DenseVector[Double] = DenseVector[Double](5.0, 5.0, 5.0)
    val allTokenDistributions: DenseMatrix[Double] = new DenseMatrix[Double](5, 3,
      Array[Double](0.6, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.6, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.6))

    val documents = simulateLDAData(
      alpha,
      allTokenDistributions,
      numDocuments = 20,
      numTokensPerDocument = 1000
    )
    val documentsRDD = sc.parallelize(documents)

    val sketcher = TensorSketcher[Double, Double](
      n = Seq(3, 3, 3),
      B = 10,
      b = Math.pow(2, 2).toInt
    )

    val tensorLDA = new TensorLDASketch(
      dimK = 3,
      alpha0 = sum(alpha),
      sketcher = sketcher,
      maxIterations = 50,
      nonNegativeDocumentConcentration = false,
      randomisedSVD = false
    )

    val (fitted_beta: DenseMatrix[Double], fitted_alpha: DenseVector[Double]) = tensorLDA.fit(documentsRDD)

    info(fitted_beta.toString())
    info(fitted_alpha.toString())
  }
*/
}
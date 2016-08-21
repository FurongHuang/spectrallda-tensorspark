package edu.uci.eecs.spectralLDA.algorithm

import breeze.linalg.{*, DenseMatrix, DenseVector, SparseVector, Vector, diag, norm, sum}
import breeze.numerics.{abs, lgamma, log}
import breeze.stats.distributions.{Dirichlet, Rand, RandBasis}
import org.apache.spark.rdd.RDD


class TensorLDAModel(val topicWordDistribution: DenseMatrix[Double],
                     val alpha: DenseVector[Double])
                    (implicit smoothing: Double = 0.01)
    extends Serializable {

  assert(topicWordDistribution.cols == alpha.length)
  assert(topicWordDistribution.forall(_ > - 1e-12))
  assert(alpha.forall(_ > 1e-10))

  private val k = alpha.length
  private val vocabSize = topicWordDistribution.rows

  // smoothing so that beta is positive
  val beta: DenseMatrix[Double] = topicWordDistribution * (1 - smoothing)
  beta += DenseMatrix.ones[Double](vocabSize, k) * (smoothing / vocabSize)

  assert(sum(beta(::, *)).toDenseVector.forall(a => abs(a - 1) <= 1e-10))
  assert(beta.forall(_ > 1e-10))

  /** compute sum of loglikelihood(doc|topics over the doc, alpha, beta) */
  def logLikelihood(docs: RDD[(Long, SparseVector[Double])],
                    maxIterationsEM: Int = 3)
      : Double = {
    docs
      .map {
        case (id: Long, wordCounts: SparseVector[Double]) =>
          val topicDistribution: DenseVector[Double] = inferEM(wordCounts, maxIterationsEM)
          TensorLDAModel.multinomialLogLikelihood(beta * topicDistribution, wordCounts)
      }
      .sum
  }

  def inferEM(wordCounts: SparseVector[Double], maxIterationsEM: Int)
             (implicit randBasis: RandBasis = Rand)
      : DenseVector[Double] = {
    var prior = alpha.copy
    var topicDistributionSample: DenseVector[Double] = null

    var latentTopicAttribution: DenseMatrix[Double] = DenseMatrix.zeros[Double](vocabSize, k)
    var wordCountsPerTopic: DenseMatrix[Double] = DenseMatrix.zeros[Double](vocabSize, k)

    for (i <- 0 until maxIterationsEM) {
      topicDistributionSample = new Dirichlet(prior).sample()
      // println(s"prior $prior, sample $topicDistributionSample")

      val expectedWordCounts: DenseVector[Double] = beta * topicDistributionSample

      for (j <- 0 until k) {
        latentTopicAttribution(::, j) := beta(::, j) * topicDistributionSample(j) / expectedWordCounts
        wordCountsPerTopic(::, j) := wordCounts :* latentTopicAttribution(::, j)
      }

      val priorIncrement: DenseVector[Double] = sum(wordCountsPerTopic(::, *)).toDenseVector
      assert(abs(sum(priorIncrement) - sum(wordCounts)) <= 1e-6)

      prior += priorIncrement
    }

    topicDistributionSample
  }
}

private[algorithm] object TensorLDAModel {
  /** compute the loglikelihood of sample x for multinomial(p) */
  def multinomialLogLikelihood(p: DenseVector[Double],
                               x: Vector[Double])
      : Double = {
    assert(p.length == x.length)
    assert(p forall(_ >= 0.0))
    assert(x forall(_ >= 0.0))

    val coeff: Double = lgamma(sum(x) + 1) - sum(x.map(a => lgamma(a + 1)))
    coeff + sum(x :* log(p))
  }
}
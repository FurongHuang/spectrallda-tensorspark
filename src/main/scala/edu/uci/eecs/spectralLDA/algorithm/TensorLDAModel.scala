package edu.uci.eecs.spectralLDA.algorithm

import breeze.linalg.{*, DenseMatrix, DenseVector, SparseVector, Vector, all, diag, sum}
import breeze.numerics.{abs, lgamma, log}
import breeze.stats.distributions.Dirichlet
import org.apache.spark.rdd.RDD


class TensorLDAModel(val beta: DenseMatrix[Double],
                     val alpha: DenseVector[Double])
    extends Serializable {
  /** compute sum of loglikelihood(doc|topics over the doc, alpha, beta) */
  def logLikelihood(docs: RDD[(Long, SparseVector[Double])],
                    maxIterationsEM: Int = 3,
                    lowerBound: Double = Double.NegativeInfinity)
      : Double = {
    docs
      .map {
        case (id: Long, wordCounts: SparseVector[Double]) =>
          val topicDistribution: DenseVector[Double] = inferEM(wordCounts, maxIterationsEM)
          val ll = TensorLDAModel.multinomialLogLikelihood(beta * topicDistribution, wordCounts)
          Math.max(lowerBound, ll)
      }
      .sum
  }

  def inferEM(wordCounts: SparseVector[Double], maxIterationsEM: Int)
      : DenseVector[Double] = {
    var prior = alpha.copy
    var topicDistributionSample: DenseVector[Double] = null

    for (i <- 0 until maxIterationsEM) {
      topicDistributionSample = new Dirichlet(prior).sample()
      // println(s"prior $prior, sample $topicDistributionSample")

      val expectedWordCounts: DenseVector[Double] = beta * topicDistributionSample
      val latentTopicAttribution: DenseMatrix[Double] =
        diag(1.0 / expectedWordCounts) * (beta * diag(topicDistributionSample))

      val wordCountsPerTopic = diag(wordCounts) * latentTopicAttribution
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
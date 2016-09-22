package edu.uci.eecs.spectralLDA.algorithm

import breeze.linalg.{*, DenseMatrix, DenseVector, SparseVector, Vector, any, max, sum}
import breeze.numerics._
import breeze.stats.distributions.Gamma
import org.apache.spark.rdd.RDD


class TensorLDAModel(val topicWordDistribution: DenseMatrix[Double],
                     val alpha: DenseVector[Double])
    extends Serializable {

  assert(topicWordDistribution.cols == alpha.length)
  assert(topicWordDistribution.forall(_ > - 1e-12))
  assert(alpha.forall(_ > 1e-10))

  private val k = alpha.length
  private val vocabSize = topicWordDistribution.rows

  /** Compute sum of loglikelihood(doc|topics over the doc, alpha, beta) */
  def logLikelihood(docs: RDD[(Long, SparseVector[Double])],
                    smoothing: Double,
                    maxIterations: Int)
      : Double = {
    val smoothedBeta = smoothBeta(docs, smoothing)
    logLikelihoodBound(docs, smoothedBeta, gammaShape = 1.0, maxIterations = maxIterations)
  }

  def variationalTopicInference(termCounts: SparseVector[Double],
                                beta: DenseMatrix[Double],
                                gammaShape: Double,
                                maxIterations: Int)
      : DenseVector[Double] = {
    // Initialize the variational distribution q(theta|gamma) for the mini-batch
    val gammad = new Gamma(gammaShape, 1.0 / gammaShape).samplesVector(k)        // K
    val expElogthetad: DenseVector[Double] = exp(TensorLDAModel.dirichletExpectation(gammad))  // K
    val expElogbetad = beta(termCounts.activeKeysIterator.toSeq, ::).toDenseMatrix    // ids * K

    val phiNorm: DenseVector[Double] = expElogbetad * expElogthetad :+ 1e-100            // ids
    var meanGammaChange = 1D
    val ctsVector = DenseVector[Double](termCounts.activeValuesIterator.toSeq: _*)                                         // ids

    // Iterate between gamma and phi until convergence
    var iter = 0
    while (iter < maxIterations && meanGammaChange > 1e-3) {
      val lastgamma = gammad.copy
      //        K                  K * ids               ids
      gammad := (expElogthetad :* (expElogbetad.t * (ctsVector :/ phiNorm))) :+ alpha
      expElogthetad := exp(TensorLDAModel.dirichletExpectation(gammad))
      // TODO: Keep more values in log space, and only exponentiate when needed.
      phiNorm := expElogbetad * expElogthetad :+ 1e-100
      meanGammaChange = sum(abs(gammad - lastgamma)) / k

      iter += 1
    }

    gammad
  }

  def logLikelihoodBound(documents: RDD[(Long, SparseVector[Double])],
                         beta: DenseMatrix[Double],
                         gammaShape: Double,
                         maxIterations: Int): Double = {
    // transpose because dirichletExpectation normalizes by row and we need to normalize
    // by topic (columns of lambda)
    val Elogbeta: DenseMatrix[Double] = log(beta)
    val ElogbetaBc = documents.sparkContext.broadcast(Elogbeta)

    // Sum bound components for each document:
    //  component for prob(tokens) + component for prob(document-topic distribution)
    val corpusPart = documents
      .filter(_._2.activeSize > 0)
      .map {
        case (id: Long, termCounts: SparseVector[Double]) =>
          val localElogbeta = ElogbetaBc.value
          var docBound = 0.0D
          val gammad: DenseVector[Double] = variationalTopicInference(
            termCounts, beta, gammaShape, maxIterations)
          val Elogthetad: DenseVector[Double] = TensorLDAModel.dirichletExpectation(gammad)

          // E[log p(doc | theta, beta)]
          termCounts.foreachPair { case (idx, count) =>
            if (any(localElogbeta(idx, ::).t :> -20.0)) {
              docBound += count * TensorLDAModel.logSumExp(Elogthetad + localElogbeta(idx, ::).t)
            }
          }

          // E[log p(theta | alpha) - log q(theta | gamma)]
          docBound += sum((alpha - gammad) :* Elogthetad)
          docBound += sum(lgamma(gammad) - lgamma(alpha))
          docBound += lgamma(sum(alpha)) - lgamma(sum(gammad))

          docBound
      }
      .sum()

    corpusPart
  }


  def smoothBeta(docs: RDD[(Long, SparseVector[Double])],
                 smoothing: Double = 0.01)
      : DenseMatrix[Double] = {
    // smoothing so that beta is positive

    val smoothedBeta: DenseMatrix[Double] = topicWordDistribution * (1 - smoothing)
    smoothedBeta += DenseMatrix.ones[Double](vocabSize, k) * (smoothing / vocabSize)

    assert(sum(smoothedBeta(::, *)).toDenseVector.forall(a => abs(a - 1) <= 1e-10))
    assert(smoothing < 1e-4 || smoothedBeta.forall(_ > 1e-10))

    smoothedBeta
  }
}

private[algorithm] object TensorLDAModel {
  /** compute the loglikelihood of sample x for multinomial(p) */
  def multinomialLogLikelihood(p: DenseVector[Double],
                               x: Vector[Double])
      : Double = {
    assert(p.length == x.length)
    assert(p forall(_ > - 1e-12))
    assert(x forall(_ > - 1e-12))

    val coeff: Double = lgamma(sum(x) + 1) - sum(x.map(a => lgamma(a + 1)))
    coeff + sum(x :* log(p))
  }

  def dirichletExpectation(alpha: DenseVector[Double]): DenseVector[Double] = {
    digamma(alpha) - digamma(sum(alpha))
  }

  def logSumExp(x: DenseVector[Double]): Double = {
    val a = max(x)
    a + log(sum(exp(x :- a)))
  }
}
package edu.uci.eecs.spectralLDA

import org.apache.spark.{SparkConf, SparkContext}
import edu.uci.eecs.spectralLDA.algorithm._
import org.apache.spark.rdd._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg._

object CVLogPerplexity {
  def main(args: Array[String]) = {
    val conf: SparkConf = new SparkConf().setAppName(s"Spectral LDA")
    val sc: SparkContext = new SparkContext(conf)

    val cv = args(0).toInt
    val documentsPath = args(1)
    val k = args(2).toInt
    val alpha0 = args(3).toDouble

    val docs = sc.objectFile[(Long, breeze.linalg.SparseVector[Double])](documentsPath)

    for (i <- 0 until cv) {
      val splits = docs.randomSplit(Array[Double](0.9, 0.1))
      computeLogLikelihood(splits, k, alpha0)
    }
  }

  def computeLogLikelihood(splits: Array[RDD[(Long, breeze.linalg.SparseVector[Double])]],
                           k: Int,
                           alpha0: Double
                          ): Unit = {
    val numTestTokens = splits(1)
      .map {
        case (_, tc) => breeze.linalg.sum(tc)
      }
      .reduce(_ + _)

    val tensorLDA = new TensorLDA(dimK = k, alpha0 = alpha0)
    val (beta, alpha, _, _, m1) = tensorLDA.fit(splits(0))

    val augBeta = breeze.linalg.DenseMatrix.zeros[Double](beta.rows, k + 1)
    val augAlpha = breeze.linalg.DenseVector.ones[Double](alpha.length + 1)
    augBeta(::, 0 until k) := beta
    augBeta(::, k) := m1
    augAlpha(0 until k) := alpha

    val tensorLDAModel = new TensorLDAModel(augBeta, augAlpha)
    val tensorLDALogL = tensorLDAModel.logLikelihood(splits(1), smoothing = 1e-6, maxIterations = 50)
    println(s"Tensor LDA log-perplexity no extra smoothing: ${- tensorLDALogL / numTestTokens}")

    val trainMapped: RDD[(Long, Vector)] = splits(0).map {
      case (id, tc) =>
        val (idx, v) = tc.activeIterator.toArray.unzip
        (id, new SparseVector(tc.length, idx, v))
    }

    val testMapped: RDD[(Long, Vector)] = splits(1).map {
      case (id, tc) =>
        val (idx, v) = tc.activeIterator.toArray.unzip
        (id, new SparseVector(tc.length, idx, v))
    }

    val ldaOptimizer = new OnlineLDAOptimizer()
      .setMiniBatchFraction(0.05)
    val lda = new LDA()
      .setOptimizer(ldaOptimizer)
      .setMaxIterations(80)
      .setK(k)
      .setDocConcentration(alpha0 / k.toDouble)

    val ldaModel: LDAModel = lda.run(trainMapped)
    val ldaLogL = ldaModel.asInstanceOf[LocalLDAModel].logLikelihood(testMapped)

    println(s"Variational Inference log-perplexity: ${- ldaLogL / numTestTokens}")
  }
}
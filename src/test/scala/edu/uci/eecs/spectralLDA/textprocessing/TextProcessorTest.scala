package edu.uci.eecs.spectralLDA.textprocessing

import org.scalatest._
import org.apache.spark.SparkContext
import edu.uci.eecs.spectralLDA.testharness.Context
import breeze.linalg._

class TextProcessorTest extends FlatSpec with Matchers {

  private val sc: SparkContext = Context.getSparkContext

  "Inverse Document Frequency" should "be correct" in {
    val docs = Seq[(Long, SparseVector[Double])](
      (1000L, SparseVector[Double](0.0, 0.0, 7.0)),
      (1001L, SparseVector[Double](0.0, 3.0, 5.0)),
      (1002L, SparseVector[Double](1.0, 3.0, 0.5))
    )
    val docsRDD = sc.parallelize[(Long, SparseVector[Double])](docs)

    val idf = TextProcessor.inverseDocumentFrequency(docsRDD)

    norm(idf - DenseVector[Double](3.0, 1.5, 1.0)) should be <= 1e-8
  }

  "Filtering for a given IDF lower bound" should "be correct" in {
    val docs = Seq[(Long, SparseVector[Double])](
      (1000L, SparseVector[Double](0.0, 0.0, 7.0)),
      (1001L, SparseVector[Double](0.0, 3.0, 5.0)),
      (1002L, SparseVector[Double](1.0, 3.0, 0.5))
    )
    val docsRDD = sc.parallelize[(Long, SparseVector[Double])](docs)

    val filteredRDD = TextProcessor.filterIDF(docsRDD, 1.5)

    filteredRDD.collect should contain theSameElementsAs Array[(Long, SparseVector[Double])](
      (1000L, SparseVector[Double](0.0, 0.0, 0.0)),
      (1001L, SparseVector[Double](0.0, 3.0, 0.0)),
      (1002L, SparseVector[Double](1.0, 3.0, 0.0))
    )
  }
}
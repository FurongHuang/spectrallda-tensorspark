package Algorithm


/**
 * Tensor Decomposition Algorithms.
 * Alternating Least Square algorithm is implemented.
 * Created by Furong Huang on 11/2/15.
 */
import DataMoments.DataCumulant
import TextProcess.SimpleTokenizer
import Utils.AlgebraUtil
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.util.control.Breaks._

class TensorLDA(sc:SparkContext, slices_string: String, paths: Seq[String], stopwordFile: String, synthetic: Int, vocabSize: Int, dimK: Int,alpha0: Double, tolerance: Double) extends Serializable{
  private val slices:Int = slices_string.toInt
  println("Start reading data...")
  val (documents: RDD[(Long, Double, SparseVector[Double])], vocabArray: Array[String], dimVocab: Int) = if (synthetic == 1) {processDocuments_synthetic(paths, vocabSize)} else { processDocuments(paths, stopwordFile, vocabSize)}
  val numDocs: Long = documents.count()
  println("Finished reading data.")
  private val myData: DataCumulant = new DataCumulant(sc, slices, dimK, alpha0, tolerance, documents,dimVocab,numDocs)

  def runALS(maxIterations: Int): (DenseMatrix[Double], DenseVector[Double])={
    val myALS: ALS = new ALS(slices, dimK, myData)
    myALS.run(sc, maxIterations)
  }




  private def processDocuments(paths: Seq[String], stopwordFile: String, vocabSize: Int): (RDD[(Long, Double, breeze.linalg.SparseVector[Double])], Array[String], Int) = {
    val textRDD: RDD[String] = sc.textFile(paths.mkString(","))
    println("successfully read the raw data." )
	textRDD.cache()
    // Split text into words
    val tokenizer: SimpleTokenizer = new SimpleTokenizer(sc, stopwordFile)
    val tokenized: RDD[(Long, IndexedSeq[String])] = textRDD.zipWithIndex().map { case (text, id) =>
      id -> tokenizer.getWords(text)
    }
    tokenized.cache()
    
    // Counts words: RDD[(word, wordCount)]
    val wordCounts: RDD[(String, Long)] = tokenized.flatMap{ case (_, tokens) => tokens.map(_ -> 1L) }.reduceByKey(_ + _)
    wordCounts.cache()
    val fullVocabSize: Long = wordCounts.count()
    
	
	
    // Select vocab
    val (vocab: Map[String, Int], selectedTokenCount: Long) = {
      val tmpSortedWC: Array[(String, Long)] = if (vocabSize == -1 || fullVocabSize <= vocabSize) {
        // Use all terms
        wordCounts.collect().sortBy(-_._2)
      } else {
        // Sort terms to select vocab
        wordCounts.sortBy(_._2, ascending = false).take(vocabSize)
      }
      (tmpSortedWC.map(_._1).zipWithIndex.toMap, tmpSortedWC.map(_._2).sum)
    }
    println("selectedTokenCount: " + selectedTokenCount.toString)
    println("vocab.size: " + vocab.size.toString)

    val mydocuments: RDD[(Long, Double, breeze.linalg.SparseVector[Double])] = tokenized.map { case (id, tokens) =>
      // Filter tokens by vocabulary, and create word count vector representation of document.
      val wc: mutable.HashMap[Int, Int] = new mutable.HashMap[Int, Int]()
      tokens.foreach { term =>
        if (vocab.contains(term)) {
          val termIndex: Int = vocab(term)
          wc(termIndex) = wc.getOrElse(termIndex, 0) + 1
        }
      }
      val indices: Array[Int] = wc.keys.toArray.sorted
      val values: Array[Double] = indices.map(i => wc(i).toDouble)
      val len: Double = values.sum
      // values = values.map(x => x/len)
      // val sb: Vector = Vectors.sparse(vocab.size, indices, values)
      val sb: breeze.linalg.SparseVector[Double] = {
        new breeze.linalg.SparseVector[Double](indices, values, vocab.size)
      }
      (id, len, sb)
    }
    println("successfully got the documents.")
    val vocabarray: Array[String] = new Array[String](vocab.size)
    vocab.foreach { case (term, i) => vocabarray(i) = term }

    (mydocuments, vocabarray, vocab.size)
  }

  private def processDocuments_synthetic(paths: Seq[String], vocabSize: Int): (RDD[(Long, Double, SparseVector[Double])], Array[String], Int) ={
    val mypath: String = paths.mkString(",")
    println(mypath)
    val mylabeledpoints: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, mypath)
    val mydocuments: RDD[(Long, Double, breeze.linalg.SparseVector[Double])] = mylabeledpoints.map(f => (f.label.toLong, f.features.toArray.sum, new breeze.linalg.SparseVector[Double](f.features.toSparse.indices, f.features.toSparse.values, f.features.toSparse.size)))
    // val mydocuments_collected: Array[(Long, Double, SparseVector[Double])] = mydocuments.collect()
    val vocabsize = mydocuments.collect()(0)._3.length
    val vocabarray: Array[String] = (0 until vocabsize).toArray.map(x => x.toString)
    (mydocuments, vocabarray, vocabsize)
  }

}

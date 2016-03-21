/**
 * Learning LDA model through inverse method of moments (moment matching).
 * Decomposition of third order moment (tensor) finds the model parameter.
 * Created by Furong Huang on 11/2/15.
 */

package edu.uci.eecs.spectralLDA

import edu.uci.eecs.spectralLDA.algorithm.TensorLDA
import edu.uci.eecs.spectralLDA.parameterParser.AbstractParams
import breeze.linalg.{DenseVector, DenseMatrix, SparseVector}
import org.apache.spark.{SparkConf,SparkContext}
import scopt.OptionParser
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import java.io._

object SpectralLDA {
  private case class Params(
                             input: Seq[String] = Seq.empty,//use this for customized input
                             libsvm: Int = 1, //Note: for real texts set, set parameter "libsvm"=0 (default), use real text (per article per row) 
                             slices: String = "2",
                             k: Int = 20,
                             maxIterations: Int = 10,
                             tolerance: Double = 1e-9,
                             topicConcentration: Double = 0.001,
                             vocabSize: Int = -1,
                             stopWordFile: String ="src/main/resources/Data/datasets/StopWords_common.txt"
                          ) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser: OptionParser[Params] = new OptionParser[Params]("LDA Example") {
      head("Tensor Factorization Step 1: reading corpus from plain text data.")
      opt[String]("slices")
        .text(s"number of workers. default: ${defaultParams.slices}")
        .action((x, c) => c.copy(slices = x))
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[Double]("tolerance")
        .text(s"tolerance. default: ${defaultParams.tolerance}")
        .action((x, c) => c.copy(tolerance = x))
      opt[Double]("topicConcentration")
        .text("amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
        s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[Int]("vocabSize")
        .text("number of distinct word types to use, chosen by frequency. (-1=all)" +
        s"  default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
      opt[Int]("libsvm")
        .text("whether to use libsvm data or real text (0=real text, 1=libsvm data)" +
        s"  default:${defaultParams.libsvm}")
        .action((x, c) => c.copy(libsvm = x))
      opt[String]("stopWordFile")
        .text("filepath for a list of stopwords. Note: This must fit on a single machine." +
        s"  default: ${defaultParams.stopWordFile}")
        .action((x, c) => c.copy(stopWordFile = x))
      arg[String]("<input>...")
        .text("input paths (directories) to plain text corpora." +
        "  Each text file line should hold 1 document.")
        .unbounded()
        .required()
        .action((x, c) => c.copy(input = c.input :+ x))
    }

	val applicationStart: Long = System.nanoTime()
    val (corpus: Array[(Long, Double, SparseVector[Double])], vocabArray: Array[String], beta: DenseMatrix[Double], alpha:DenseVector[Double]) = parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      parser.showUsageAsError
      sys.exit(1)
    }
    
    val applicationElapsed: Double = (System.nanoTime() - applicationStart) / 1e9
    
    {
	  //time
	  val writer_time = new PrintWriter(new File("runningTime.txt"))
      writer_time.write(s"\t application running time: $applicationElapsed sec")
      writer_time.close()
      
      
      println()
      println("Learning done. Writing topic word matrix (beta) and topic proportions (alpha)... ")
      // beta
      breeze.linalg.csvwrite(new File("beta.txt"), beta, separator = ' ')
      //alpha
      // println(alpha.map(x => math.abs(x/alpha0Estimate*defaultParams.topicConcentration)))
      val alpha0Estimate:Double = breeze.linalg.sum(alpha)
      val writer_alpha = new PrintWriter(new File("alpha.txt" ))      
      var i = 0
      for( i <- 0 to alpha.length-1){
         writer_alpha.write(s"$alpha(i)/alpha0Estimate*defaultParams.topicConcentration \t")
      }
      writer_alpha.close()
      
      
    }
  }


  private def run(params: Params): (Array[(Long, Double, SparseVector[Double])], Array[String], DenseMatrix[Double], DenseVector[Double]) = {

    Logger.getRootLogger.setLevel(Level.WARN)
    if (params.libsvm == 1) {
      println("Input data in libsvm format.")
    }
    else {
      println("Converting raw text to libsvm format.")
    }
    val preprocessStart: Long = System.nanoTime()
    val conf: SparkConf = new SparkConf().setAppName("Spectral LDA via Tensor Decomposition")
    val sc: SparkContext = new SparkContext(conf)
        println("Generated the SparkConetxt")
    val myTensorLDA: TensorLDA = new TensorLDA(sc, params.input, params.stopWordFile, params.libsvm, params.vocabSize, params.k, params.topicConcentration, params.tolerance)
    println("Start ALS algorithm for tensor decomposition...")

    val (beta, alpha) = myTensorLDA.runALS(params.maxIterations)
    println("Finished ALS algorithm for tensor decomposition.")

    val numDocs: Long = myTensorLDA.numDocs
    val vocabSize: Int = myTensorLDA.dimVocab
    val corpus = myTensorLDA.documents
    val vocabArray = myTensorLDA.vocabArray
    val corpus_collection: Array[(Long, Double, breeze.linalg.SparseVector[Double])] = corpus.collect()
    val preprocessElapsed: Double = (System.nanoTime() - preprocessStart) / 1e9
    sc.stop()
    println()
    println("Corpus summary:")
    println(s"\t Training set size: $numDocs documents")
    println(s"\t Vocabulary size: $vocabSize terms")
    println(s"\t Model Training time: $preprocessElapsed sec")
    println()
    (corpus_collection, vocabArray, beta, alpha)
  }

}

 /**
 * Learning LDA model through inverse method of moments (moment matching).
 * Decomposition of third order moment (tensor) finds the model parameter.
 * Created by Furong Huang on 11/2/15.
 */

package edu.uci.eecs.spectralLDA

import edu.uci.eecs.spectralLDA.algorithm.TensorLDA
import edu.uci.eecs.spectralLDA.textprocessing.TextProcessor
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
                             k: Int = 20,
                             maxIterations: Int = 1000,
                             tolerance: Double = 1e-9,
                             topicConcentration: Double = 0.001,
                             vocabSize: Int = -1,
                             stopWordFile: String ="src/main/resources/Data/datasets/StopWords_common.txt"
                          )

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser: OptionParser[Params] = new OptionParser[Params]("LDA Example") {
      head("Tensor Factorization Step 1: reading corpus from plain text data.")
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

    val (corpus: RDD[(Long, SparseVector[Double])], vocabArray: Array[String], beta: DenseMatrix[Double], alpha:DenseVector[Double]) = parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      parser.showUsageAsError
      sys.exit(1)
    }
  }


  private def run(params: Params): (RDD[(Long, SparseVector[Double])], Array[String], DenseMatrix[Double], DenseVector[Double]) = {

    Logger.getRootLogger.setLevel(Level.WARN)
    if (params.libsvm == 1) {
      println("Input data in libsvm format.")
    }
    else {
      println("Converting raw text to libsvm format.")
    }

    val applicationStart: Long = System.nanoTime()
    val preprocessStart: Long = System.nanoTime()

    val conf: SparkConf = new SparkConf().setAppName(s"Spectral LDA via Tensor Decomposition: $params")
    val sc: SparkContext = new SparkContext(conf)
    println("Generated the SparkConetxt")

    println("Start reading data...")
    val (documents: RDD[(Long, SparseVector[Double])], vocabArray: Array[String]) = if (params.libsvm == 1) {
      TextProcessor.processDocuments_libsvm(sc, params.input, params.vocabSize)
    }
    else {
      TextProcessor.processDocuments(sc, params.input, params.stopWordFile, params.vocabSize)
    }
    println("Finished reading data.")

    val myTensorLDA: TensorLDA = new TensorLDA(
      params.k,
      params.topicConcentration,
      params.maxIterations,
      params.tolerance
    )
    println("Start ALS algorithm for tensor decomposition...")

    val (beta, alpha) = myTensorLDA.fit(documents)
    println("Finished ALS algorithm for tensor decomposition.")

    val preprocessElapsed: Double = (System.nanoTime() - preprocessStart) / 1e9
    val numDocs: Long = documents.count()
    val dimVocab: Int = documents.take(1)(0)._2.length
    sc.stop()
    println()
    println("Corpus summary:")
    println(s"\t Training set size: $numDocs documents")
    println(s"\t Vocabulary size: $dimVocab terms")
    println(s"\t Model Training time: $preprocessElapsed sec")
    println()

    //time
    val applicationElapsed: Double = (System.nanoTime() - applicationStart) / 1e9
    val writer_time = new PrintWriter(new File(s"runningTime.txt"))
    writer_time.write(s"$applicationElapsed sec")
    writer_time.close()

    println()
    println("Learning done. Writing topic word matrix (beta) and topic proportions (alpha)... ")

    // beta
    breeze.linalg.csvwrite(new File(s"beta.txt"), beta, separator = ' ')

    //alpha
    // println(alpha.map(x => math.abs(x/alpha0Estimate*params.topicConcentration)))
    val alpha0Estimate:Double = breeze.linalg.sum(alpha)
    val writer_alpha = new PrintWriter(new File(s"alpha.txt" ))
    var i = 0
    for( i <- 0 until alpha.length){
      var thisAlpha: Double = alpha(i) / alpha0Estimate * params.topicConcentration
      writer_alpha.write(s"$thisAlpha \t")
    }
    writer_alpha.close()
    (documents, vocabArray, beta, alpha)
  }
}


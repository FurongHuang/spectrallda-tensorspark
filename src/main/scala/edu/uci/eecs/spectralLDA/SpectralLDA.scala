 /**
 * Learning LDA model through inverse method of moments (moment matching).
 * Decomposition of third order moment (tensor) finds the model parameter.
 * Created by Furong Huang on 11/2/15.
 */

package edu.uci.eecs.spectralLDA

import edu.uci.eecs.spectralLDA.algorithm.{TensorLDA, TensorLDASketch}
import edu.uci.eecs.spectralLDA.textprocessing.TextProcessor
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector}
import org.apache.spark.{SparkConf, SparkContext}

import scalaxy.loops._
import scala.language.postfixOps
import scopt.OptionParser
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import java.io._

import edu.uci.eecs.spectralLDA.sketch.TensorSketcher

object SpectralLDA {
  private case class Params(
                             input: Seq[String] = Seq.empty,//use this for customized input
                             inputType: String = "obj", // "libsvm", "text" or "obj"
                             k: Int = 1,
                             topicConcentration: Double = 5.0,
                             maxIterations: Int = 200,
                             tolerance: Double = 1e-9,
                             vocabSize: Int = -1,
                             sketching: Boolean = false,
                             B: Int = 50,
                             b: Int = Math.pow(2, 8).toInt,
                             outputDir: String = ".",
                             stopWordFile: String = "src/main/resources/Data/datasets/StopWords_common.txt"
                          )

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser: OptionParser[Params] = new OptionParser[Params]("LDA Example") {
      head("Spectral LDA Factorization")

      opt[Int]('k', "k").required()
        .text("number of topics")
        .action((x, c) => c.copy(k = x))
        .validate(x =>
          if (x > 0) success
          else failure("The number of topics k must be positive.")
        )
      opt[Double]("topicConcentration").abbr("alpha0").required()
        .text("the sum of the prior vector for topic distribution e.g. 5k or 10k. The higher it is; the less variation in the drawn topic distribution vector")
        .action((x, c) => c.copy(topicConcentration = x))
        .validate(x =>
          if (x > 0.0) success
          else failure("topicConcentration must be positive.")
        )

      opt[Int]("maxIterations").abbr("max-iter")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
        .validate(x =>
          if (x > 0) success
          else failure("maxIterations must be positive.")
        )
      opt[Double]("tolerance").abbr("tol")
        .text(s"tolerance. default: ${defaultParams.tolerance}")
        .action((x, c) => c.copy(tolerance = x))
        .validate(x =>
          if (x > 0.0) success
          else failure("tolerance must be positive.")
        )

      opt[Int]('V', "vocabSize")
        .text(s"number of distinct word types to use, ordered by frequency. default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
        .validate(x =>
          if (x == -1 || x > 0) success
          else failure("vocabSize must be -1 for all or positive."))

      opt[String]('t', "inputType")
        .text(s"""type of input files: "obj", "libsvm" or "text". "obj" for Hadoop SequenceFile of RDD[(Long, SparseVector[Double])]. default: ${defaultParams.inputType}""")
        .action((x, c) => c.copy(inputType = x))
        .validate(x =>
          if (x == "obj" || x == "libsvm" || x == "text") success
          else failure("""inputType must be "obj", "libsvm" or "text".""")
        )

      opt[Unit]("sketching")
        .text("Tensor decomposition via sketching")
        .action((_, c) => c.copy(sketching = true))
      opt[Int]('B', "B")
        .text(s"number of hash families for sketching. default: ${defaultParams.B}")
        .action((x, c) => c.copy(B = x))
        .validate(x =>
          if (x > 0) success
          else failure("The number of hash families B for sketching must be positive.")
        )
      opt[Int]('b', "b")
        .text(s"length of a hash for sketching, preferably to be power of 2. default: ${defaultParams.b}")
        .action((x, c) => c.copy(b = x))
        .validate(x =>
          if (x > 0) success
          else failure("The length of a hash b for sketching must be positive.")
        )

      opt[String]('o', "outputDir").valueName("<dir>")
        .text(s"output write path. default: ${defaultParams.outputDir}")
        .action((x, c) => c.copy(outputDir = x))
      opt[String]("stopWordFile")
        .text(s"filepath for a list of stopwords. default: ${defaultParams.stopWordFile}")
        .action((x, c) => c.copy(stopWordFile = x))

      help("help").text("prints this usage text")

      arg[String]("<input>...")
        .text("paths of input files")
        .unbounded()
        .required()
        .action((x, c) => c.copy(input = c.input :+ x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        run(params)
      case None =>
        parser.showUsageAsError
        sys.exit(1)
    }
  }


  private def run(params: Params): (RDD[(Long, SparseVector[Double])], Array[String], DenseMatrix[Double], DenseVector[Double]) = {

    Logger.getRootLogger.setLevel(Level.WARN)
    if (params.inputType == "libsvm") {
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
    val (documents: RDD[(Long, SparseVector[Double])], vocabArray: Array[String]) = params.inputType match {
      case "libsvm" =>
        TextProcessor.processDocuments_libsvm(sc, params.input.mkString(","), params.vocabSize)
      case "text" =>
        TextProcessor.processDocuments(sc, params.input.mkString(","), params.stopWordFile, params.vocabSize)
      case "obj" =>
        (sc.objectFile(params.input.mkString(",")), null)
    }
    println("Finished reading data.")

    println("Start ALS algorithm for tensor decomposition...")
    val (beta, alpha) = if (params.sketching) {
      println("Running tensor decomposition via sketching...")
      val sketcher = TensorSketcher[Double, Double](
        n = Seq(params.k, params.k, params.k),
        B = params.B,
        b = params.b
      )
      val lda = new TensorLDASketch(
        dimK = params.k,
        alpha0 = params.topicConcentration,
        sketcher = sketcher,
        maxIterations = params.maxIterations,
        nonNegativeDocumentConcentration = true,
        randomisedSVD = true
      )(tolerance = params.tolerance)
      lda.fit(documents)
    }
    else {
      val lda = new TensorLDA(
        params.k,
        params.topicConcentration,
        params.maxIterations,
        params.tolerance
      )
      lda.fit(documents)
    }
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
    val thisK = params.k
    val applicationElapsed: Double = (System.nanoTime() - applicationStart) / 1e9
    val writer_time = new PrintWriter(new File(params.outputDir + s"/TD_runningTime_k$thisK" + ".txt"))
    writer_time.write(s"$applicationElapsed sec")
    writer_time.close()

    println()
    println("Learning done. Writing topic word matrix (beta) and topic proportions (alpha)... ")

    // beta
    breeze.linalg.csvwrite(new File(params.outputDir + s"/TD_beta_k$thisK" + ".txt"), beta, separator = ' ')

    //alpha
    // println(alpha.map(x => math.abs(x/alpha0Estimate*params.topicConcentration)))
    val alpha0Estimate:Double = breeze.linalg.sum(alpha)
    val writer_alpha = new PrintWriter(new File(params.outputDir + s"/alpha_k$thisK" + ".txt" ))
    for( i <- 0 until alpha.length optimized){
      var thisAlpha: Double = alpha(i) / alpha0Estimate * params.topicConcentration
      writer_alpha.write(s"$thisAlpha \t")
    }
    writer_alpha.close()

    (documents, vocabArray, beta, alpha)
  }
}

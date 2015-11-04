/**
 * Learning LDA model through inverse method of moments (moment matching).
 * Decomposition of third order moment (tensor) finds the model parameter.
 * Created by Furong Huang on 11/2/15.
 */

import Algorithm.TensorLDA
import ParameterParser.AbstractParams
import breeze.linalg.{DenseVector, DenseMatrix, SparseVector}
import org.apache.spark.{SparkConf,SparkContext}
import scopt.OptionParser
import org.apache.log4j.{Level, Logger}

object SpectralLDA {
  private case class Params(
                             input: Seq[String] = Seq.empty,//use this for customized input
                             synthetic: Int = 0, //Note: for real texts set, set parameter "synthetic"=0 (default), use real text (per article per row) "SpectralLDA-Tensor/src/main/resources/Data/datasets/enron_email/corpus.txt"
                             // synthetic: Int = 1, //Note: for synthetic data, set parameter "synthetic"=1, use synthetic data "SpectralLDA-Tensor/src/main/resources/Data/datasets/synthetic/samples_train_libsvm.txt"
                             slices: String = "2",
                             k: Int = 2,
                             maxIterations: Int = 100,
                             tolerance: Double = 1e-9,
                             topicConcentration: Double = 0.001,
                             vocabSize: Int = -1,
                             stopWordFile: String = s"src/main/resources/Data/datasets/StopWords_common.txt",
                             reducerMaxSizeInFlight:String = "1g",
                             executorMemory:String ="1g",
                             driveMemory:String ="1g",
                             shuffleFileBuffer:String ="1g",
                             driverMaxResultSize:String ="1g",
                             storageMemoryFraction:String ="0.5",
                             sparkRddCompress:String ="true") extends AbstractParams[Params]

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
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
        s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[Int]("vocabSize")
        .text(s"number of distinct word types to use, chosen by frequency. (-1=all)" +
        s"  default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
      opt[Int]("synthetic")
        .text(s"whether to use synthetic data or real text (0=real text, 1=synthetic data)" +
        s"  default:${defaultParams.synthetic}")
        .action((x, c) => c.copy(synthetic = x))
      opt[String]("stopWordFile")
        .text(s"filepath for a list of stopwords. Note: This must fit on a single machine." +
        s"  default: ${defaultParams.stopWordFile}")
        .action((x, c) => c.copy(stopWordFile = x))
      arg[String]("<input>...")
        .text("input paths (directories) to plain text corpora." +
        "  Each text file line should hold 1 document.")
        .unbounded()
        .required()
        .action((x, c) => c.copy(input = c.input :+ x))
    }

    val (corpus: Array[(Long, Double, SparseVector[Double])], vocabArray: Array[String], beta: DenseMatrix[Double], alpha:DenseVector[Double]) = parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      parser.showUsageAsError
      sys.exit(1)
    }

    {
      println()
      println("topic word matrix (beta): ")
      println(beta)

      println()
      println("topic proportions (alpa): ")
      val alpha0Estimate:Double = breeze.linalg.sum(alpha)
      println(alpha.map(x => math.abs(x/alpha0Estimate*defaultParams.topicConcentration)))

      println()
      println("corpus: ")
      println(corpus)

      println()
      println("vocabulary: ")
      println(vocabArray)
    }
  }


  private def run(params: Params): (Array[(Long, Double, SparseVector[Double])], Array[String], DenseMatrix[Double], DenseVector[Double]) = {

    Logger.getRootLogger.setLevel(Level.WARN)
    if (params.synthetic == 1) {
      println("Running synthetic example.")
    }
    else {
      println("Running real example.")
    }
    val preprocessStart: Long = System.nanoTime()
	println("Generating a new SparkContext with "+ params.slices +" slices... ")
    val conf: SparkConf = new SparkConf().setMaster("local" + "[" + params.slices + "]").set("spark.reducer.maxSizeInFlight",params.reducerMaxSizeInFlight).set("spark.executor.memory",params.executorMemory).set("spark.driver.memory",params.driveMemory).set("spark.driver.maxResultSize",params.driverMaxResultSize).set("spark.shuffle.file.buffer",params.shuffleFileBuffer).setAppName(s"Spectral LDA via Tensor Decomposition").set("spark.storage.memoryFraction",params.storageMemoryFraction).set("spark.rdd.compress",params.sparkRddCompress)
    val sc: SparkContext = new SparkContext(conf)
	println("Generated the SparkConetxt")
    val myTensorLDA: TensorLDA = new TensorLDA(sc, params.slices, //params.reducerMaxSizeInFlight, params.executorMemory,params.driveMemory,params.driverMaxResultSize,params.shuffleFileBuffer,params.storageMemoryFraction,params.sparkRddCompress,
      params.input, params.stopWordFile, params.synthetic, params.vocabSize, params.k, params.topicConcentration, params.tolerance)
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
    println(s"Corpus summary:")
    println(s"\t Training set size: $numDocs documents")
    println(s"\t Vocabulary size: $vocabSize terms")
    println(s"\t Model Training time: $preprocessElapsed sec")
    println()
    (corpus_collection, vocabArray, beta, alpha)
  }

}

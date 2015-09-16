package LDATensorDecomposition
/**
 * Created by furongh on 9/1/15.
 furongh@uci.edu
 */
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
import org.apache.spark.broadcast.Broadcast
//import org.apache.spark.mllib.linalg
//import org.apache.spark.sql.DataFrame
import scala.util.control.Breaks._
import breeze.linalg.diag
import breeze.linalg.svd
import breeze.linalg.{DenseMatrix,DenseVector,SparseVector}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{Accumulator, SparkConf, SparkContext}
import scopt.OptionParser
import scala.collection.mutable
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
/**
 * An example Latent Dirichlet Allocation (LDA) app. Run with
 * {{{
 * ./bin/run-example mllib.LDAExample [options] <input>
 * }}}
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object SpectralLDA {

  private case class Params(
                             input: Seq[String] = Seq.empty,//use this for customized input
                             //input: Seq[String] = Seq("data/datasets/enron_email/corpus.txt"), // default example input
                             //input: Seq[String] = Seq("data/datasets/synthetic/samples_trainlibsvm.txt"), // default example input
                             slices: Int = 2,
                             k: Int = 2,
                             maxIterations: Int = 100,
                             tolerance: Double = 1e-9,
                             docConcentration: Double = -1,
                             topicConcentration: Double = 0.001,
                             vocabSize: Int = -1, //10000,
                             synthetic: Int = 1,
                             stopwordFile: String = "" //"data_topic_tiny/EnronEmailDataset/StopWords_common.txt"
                             //,algorithm: String = "em"
                             ) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser: OptionParser[Params] = new OptionParser[Params]("LDAExample") {
      head("Tensor Factorization Step 1: reading corpus from plain text data.")
      opt[Int]("slices")
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
      opt[Double]("docConcentration")
        .text(s"amount of topic smoothing to use (> 1.0) (-1=auto)." +
        s"  default: ${defaultParams.docConcentration}")
        .action((x, c) => c.copy(docConcentration = x))
      opt[Double]("topicConcentration")
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
        s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[Int]("vocabSize")
        .text(s"number of distinct word types to use, chosen by frequency. (-1=all)" +
        s"  default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
      opt[Int]("synthetic")
        .text(s"whether to use synthetic example or real example" +
        s"  default:${defaultParams.synthetic}")
        .action((x, c) => c.copy(synthetic = x))
      opt[String]("stopwordFile")
        .text(s"filepath for a list of stopwords. Note: This must fit on a single machine." +
        s"  default: ${defaultParams.stopwordFile}")
        .action((x, c) => c.copy(stopwordFile = x))
      /*opt[String]("algorithm")
        .text(s"inference algorithm to use. em and online are supported." +
        s" default: ${defaultParams.algorithm}")
        .action((x, c) => c.copy(algorithm = x))*/
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
      val alpha0Estimate:Double = alpha.sum
      println(alpha.map(x => math.abs(x/alpha0Estimate*defaultParams.topicConcentration)))
    }


  }

  private def run(params: Params): (Array[(Long, Double, SparseVector[Double])], Array[String], DenseMatrix[Double], DenseVector[Double]) = {
    val conf: SparkConf = new SparkConf()
      .setMaster("local[2]")
      .setAppName(s"Tensor Factorization Step 1: reading corpus with $params")
      .set("spark.executor.memory", "1g")
      .set("spark.rdd.compress", "true")
      .set("spark.storage.memoryFraction", "1")
    val sc: SparkContext = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)


    if (params.synthetic == 1) {
      println("Running synthetic example")
    }
    else {
      println("Running real example")
    }
    // Load documents, and prepare them for LDA.
    val preprocessStart: Long = System.nanoTime()
    val (corpus, vocabArray, actualNumTokens, beta, alpha) =
      preprocess(sc, params.synthetic, params.slices, params.input, params.vocabSize, params.k, params.maxIterations, params.tolerance, params.topicConcentration, params.stopwordFile)
    corpus.cache()
    val numDocs: Long = corpus.count()
    val vocabSize: Int = vocabArray.size
    var corpus_collection: Array[(Long, Double, breeze.linalg.SparseVector[Double])] = corpus.collect()
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

  /**
   * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
   * @return (corpus, vocabulary as array, total token count in corpus)
   */
  private def preprocess(
                          sc: SparkContext,
                          synthetic: Int,
                          slices: Int,
                          paths: Seq[String],
                          vocabSize: Int,
                          dimK: Int,
                          maxIterations: Int,
                          tolerance: Double,
                          alpha0: Double,
                          stopwordFile: String): (RDD[(Long, Double, SparseVector[Double])], Array[String], Long, breeze.linalg.DenseMatrix[Double], breeze.linalg.DenseVector[Double]) = {

    // Get dataset of document texts
    // One document per line in each text file. If the input consists of many small files,
    // this can result in a large number of small partitions, which can degrade performance.
    // In this case, consider using coalesce() to create fewer, larger partitions.
    val (documents: RDD[(Long, Double, SparseVector[Double])], vocabArray: Array[String], dimVocab: Int) =
      if (synthetic == 1) {
        processDocuments_synthetic(sc, paths, vocabSize)
      }
      else {
        processDocuments(sc, paths, stopwordFile, vocabSize)
      }


    val numDocs: Long = documents.count()

    // !!! First Order Moments
    val firstOrderMoments: breeze.linalg.SparseVector[Double] = {
      documents.flatMap(x => List(x._3.map(entry => entry / x._2))).reduce((a, b) => a :+ b)
    }.map(x => x / numDocs.toDouble)


    // !!! Third Order Moments and Unwhitening Matrix
    val (thirdOrderMoments: DenseMatrix[Double], unwhiteningMatrix: DenseMatrix[Double]) = {
      // run whiten step
      val documents_broadcasted: Broadcast[Array[(Long, Double, breeze.linalg.SparseVector[Double])]] = sc.broadcast(documents.collect())
      val (eigenVectors: DenseMatrix[Double], eigenValues: DenseVector[Double]) = whiten(sc, slices, alpha0, dimVocab, dimK, numDocs, firstOrderMoments, documents_broadcasted, tolerance)
      // println("secondOrdermoment: eigenVectors")
      // println(eigenVectors)
      // println("secondOrdermoment: eigenValues")
      // println(eigenValues)
      var Ta: Accumulator[DenseMatrix[Double]] = sc.accumulator(breeze.linalg.DenseMatrix.zeros[Double](dimK, dimK * dimK))(DenseMatrixAccumulatorParam)
      val whitenedData: RDD[(breeze.linalg.DenseVector[Double], Double)] = sc.parallelize(0 until numDocs.toInt, slices).map(i => (project(i, dimVocab, dimK, alpha0, eigenValues, eigenVectors, documents_broadcasted.value(i)._3, tolerance), documents_broadcasted.value(i)._2))
      val whitenedData_broadcasted: Broadcast[Array[(DenseVector[Double], Double)]] = sc.broadcast(whitenedData.collect())
      var firstOrderMoments_whitened: DenseVector[Double] = whitenedData.collect().map(x => x._1 / x._2).reduce((a, b) => a :+ b)
      firstOrderMoments_whitened = firstOrderMoments_whitened.map(x => x/numDocs.toDouble)

      val m1_broadcasted = sc.broadcast(firstOrderMoments_whitened)
      sc.parallelize(0 until numDocs.toInt, slices).foreach(i => Ta +=
        update_thirdOrderMoments(i, dimK, alpha0, m1_broadcasted.value, whitenedData_broadcasted.value(i)._1, whitenedData_broadcasted.value(i)._2)
      )

      val alpha0sq: Double = alpha0 * alpha0
      var Ta_shift = DenseMatrix.zeros[Double](dimK, dimK * dimK)
      for (id_i:Int <- 0 until dimK) {
        for (id_j:Int <- 0 until dimK) {
          for (id_l:Int <- 0 until dimK) {
            Ta_shift(id_i, id_j*dimK + id_l) += alpha0sq * firstOrderMoments_whitened(id_i) *  firstOrderMoments_whitened(id_j) *  firstOrderMoments_whitened(id_l)
          }
        }
      }

      //Ta -= Ta_shift


      val unwhiteningMatrix: breeze.linalg.DenseMatrix[Double] = eigenVectors * diag(eigenValues.map(x => scala.math.pow(x, 0.5)))
      (Ta.value.map(x => x/numDocs.toDouble)-Ta_shift, unwhiteningMatrix)
    }

    // println("thirdOrderMoments")
    // println(thirdOrderMoments)


    // !!! ALS iterations
    val (whitenedTopicWordMatrix: breeze.linalg.DenseMatrix[Double], lambda: breeze.linalg.DenseVector[Double]) = ALS(sc, dimK, thirdOrderMoments, maxIterations, tolerance, 2)
    // println("whitenedTopicWordMatrix")
    // println(whitenedTopicWordMatrix)

    // println("lambda")
    // println(lambda)
    // Note: alpha = lambda^(-2)
    val alpha: DenseVector[Double] = lambda.map(x => scala.math.pow(x, -2))
    // !!! unwhitening
    // R_a = U Sigma^(0.5) A Diag(alpha^(-0.5))  = U Sigma^(0.5) A Diag(lambda)
    val topicWordMatrix: breeze.linalg.DenseMatrix[Double] = unwhiteningMatrix * whitenedTopicWordMatrix * breeze.linalg.diag(lambda)
    val topicWordMatrix_normed: breeze.linalg.DenseMatrix[Double] = simplexProj_Matrix(topicWordMatrix, tolerance)

    // println("topicWordMatrix")
    // println(topicWordMatrix)

    // println("topicWordMatrix_normed")
    // println(topicWordMatrix_normed)

    return (documents, vocabArray, dimVocab, topicWordMatrix_normed, alpha)
  }


  def whiten(sc: SparkContext, slices: Int, alpha0: Double, vocabSize: Int, dimK: Int, numDocs: Long, firstOrderMoments: breeze.linalg.SparseVector[Double], documents_broadcasted: Broadcast[Array[(Long, Double, breeze.linalg.SparseVector[Double])]], tolerance: Double): (breeze.linalg.DenseMatrix[Double], breeze.linalg.DenseVector[Double]) = {
    val para_main: Double = (alpha0 + 1.0) / numDocs.toDouble
    val para_shift: Double = alpha0

    val SEED_random: Long = System.currentTimeMillis
    var gaussianRandomMatrix: DenseMatrix[Double] = gaussian(vocabSize, dimK*2, SEED_random, tolerance)
    // println(gaussianRandomMatrix.t * gaussianRandomMatrix)

    val m1_raw_broadcasted: Broadcast[breeze.linalg.DenseVector[Double]] = sc.broadcast(firstOrderMoments.toDenseVector)
    // val documents_broadcasted: Broadcast[Array[(Long, Double, breeze.linalg.SparseVector[Double])]] = sc.broadcast(documents.collect())
    var M2_a_S: Accumulator[breeze.linalg.DenseMatrix[Double]] = sc.accumulator(breeze.linalg.DenseMatrix.zeros[Double](vocabSize, dimK*2), "Second Order Moment multiplied with S: M2_a * S")(DenseMatrixAccumulatorParam)
    sc.parallelize(0 until numDocs.toInt, slices).foreach(i => M2_a_S += accumulate_M_mul_S(i, vocabSize, dimK*2, alpha0,
      m1_raw_broadcasted.value, gaussianRandomMatrix, documents_broadcasted.value(i)._3, documents_broadcasted.value(i)._2))

    M2_a_S.value *= para_main
    val shiftedMatrix: breeze.linalg.DenseMatrix[Double] = firstOrderMoments.toDenseVector * (firstOrderMoments.toDenseVector.t * gaussianRandomMatrix)
    M2_a_S.value -= shiftedMatrix :* para_shift

    val Q = orthogonalizeMatCols(M2_a_S.value, tolerance)
    var M2_a_Q: Accumulator[DenseMatrix[Double]] = sc.accumulator(breeze.linalg.DenseMatrix.zeros[Double](vocabSize, dimK*2), "Second Order Moment multiplied with S: M2_a * S")(DenseMatrixAccumulatorParam)


    sc.parallelize(0 until numDocs.toInt, slices).foreach(i => M2_a_Q += accumulate_M_mul_S(i, vocabSize, dimK*2, alpha0,
      m1_raw_broadcasted.value, Q, documents_broadcasted.value(i)._3, documents_broadcasted.value(i)._2))
    M2_a_Q.value *= para_main
    val shiftedMatrix2: breeze.linalg.DenseMatrix[Double] = firstOrderMoments.toDenseVector * (firstOrderMoments.toDenseVector.t * Q)
    M2_a_Q.value -= shiftedMatrix2 :* para_shift

    // Note: eigenvectors * Diag(eigenvalues) = M2_a_Q
    val svd.SVD(u: breeze.linalg.DenseMatrix[Double], s: breeze.linalg.DenseVector[Double], v: breeze.linalg.DenseMatrix[Double]) = svd((M2_a_Q.value.t * M2_a_Q.value))
    // val s_mat_inv: DenseMatrix[Double] = breeze.linalg.pinv(diag(s))
    val eigenVectors: DenseMatrix[Double] = M2_a_Q.value * u * diag(s.map(entry=>1.0/math.sqrt(entry)))
    // println(eigenVectors.t * eigenVectors)
    // eigenVectors = eigenVectors * breeze.linalg.pinv(diag(s))
    val eigenValues: DenseVector[Double] = s.map(entry => math.sqrt(entry))
    (eigenVectors(::,0 until dimK), eigenValues(0 until dimK))
  }

  def update_thirdOrderMoments(ii: Int, dimK: Int, alpha0: Double, m1: DenseVector[Double], Wc: DenseVector[Double], len: Double): DenseMatrix[Double] = {
    var len_calibrated: Double = 3.0
    if (len >= 3.0) {
      len_calibrated = len
    }

    val scale3fac: Double = (alpha0 + 1.0) * (alpha0 + 2.0) / (2.0 * len_calibrated * (len_calibrated - 1.0) * (len_calibrated - 2.0))
    val scale2fac: Double = alpha0 * (alpha0 + 1.0) / (2.0 * len_calibrated * (len_calibrated - 1.0))
    var Ta = breeze.linalg.DenseMatrix.zeros[Double](dimK, dimK * dimK)

    //val wc = Array(DenseVector[Double](1.0,2.0,3.0), DenseVector[Double](4.0,7.0,8.0))
    //sc.parallelize(wc).foreach { x:DenseVector[Double] =>
    for (i: Int <- 0 until dimK) {
      for (j: Int <- 0 until dimK) {
        for (l: Int <- 0 until dimK) {
          Ta(i, dimK * j + l) += scale3fac * Wc(i) * Wc(j) * Wc(l)

          Ta(i, dimK * j + l) -= scale2fac * Wc(i) * Wc(j) * m1(l)
          Ta(i, dimK * j + l) -= scale2fac * Wc(i) * m1(j) * Wc(l)
          Ta(i, dimK * j + l) -= scale2fac * m1(i) * Wc(j) * Wc(l)
        }
        Ta(i, dimK * i + j) -= scale3fac * Wc(i) * Wc(j)
        Ta(i, dimK * j + i) -= scale3fac * Wc(i) * Wc(j)
        Ta(i, dimK * j + j) -= scale3fac * Wc(i) * Wc(j)

        Ta(i, dimK * i + j) += scale2fac * Wc(i) * m1(j)
        Ta(i, dimK * j + i) += scale2fac * Wc(i) * m1(j)
        Ta(i, dimK * j + j) += scale2fac * m1(i) * Wc(j)
      }
      Ta(i, dimK * i + i) += 2.0 * scale3fac * Wc(i)
    }
    Ta
  }

  def accumulate_M_mul_S(ii: Int, dimVocab: Int, dimK: Int, alpha0: Double,
                         m1: breeze.linalg.DenseVector[Double], S: breeze.linalg.DenseMatrix[Double], Wc: breeze.linalg.SparseVector[Double], len: Double) = {
    assert(dimVocab == Wc.length)
    assert(dimVocab == m1.length)
    assert(dimVocab == S.rows)
    assert(dimK == S.cols)
    var len_calibrated: Double = 3.0
    if (len >= 3) {
      len_calibrated = len
    }

    var M2_a = breeze.linalg.DenseMatrix.zeros[Double](dimVocab, dimK)

    val norm_length: Double = 1.0 / (len_calibrated * (len_calibrated - 1.0))
    var data_mul_S: DenseVector[Double] = breeze.linalg.DenseVector.zeros[Double](dimK)

    var offset = 0
    while (offset < Wc.activeSize) {
      val token: Int = Wc.indexAt(offset)
      val count: Double = Wc.valueAt(offset)
      // val S_row = S(token,::)

      data_mul_S += S(token, ::).t.map(x => x * count)

      offset += 1
    }

    offset = 0
    while (offset < Wc.activeSize) {
      val token: Int = Wc.indexAt(offset)
      val count: Double = Wc.valueAt(offset)
      M2_a(token, ::) += (data_mul_S - S(token, ::).t).map(x => x * count * norm_length).t

      offset += 1
    }
    M2_a
  }

  def ALS(sc: SparkContext, dimK: Int, T: breeze.linalg.DenseMatrix[Double], maxIterations: Int, tolerance: Double, slices: Int): (DenseMatrix[Double], DenseVector[Double]) = {
    // val T = sc.textFile("data_topic_tiny/tensor_A.txt")
    // Iteratively update
    // val T_broadcasted = sc.broadcast(inputfile)
    val SEED_A: Long = System.currentTimeMillis
    val SEED_B: Long = System.currentTimeMillis
    val SEED_C: Long = System.currentTimeMillis
    val SEED_T: Long = System.currentTimeMillis
    //val SEED_A: Long = 976180231
    //val SEED_B: Long = 1412218460
    //val SEED_C: Long = 2048512343
    //val SEED_T: Long = 1456390857
    var A: DenseMatrix[Double] = gaussian(dimK, dimK, SEED_A, tolerance)
    var B: DenseMatrix[Double] = gaussian(dimK, dimK, SEED_B, tolerance)
    var C: DenseMatrix[Double] = gaussian(dimK, dimK, SEED_C, tolerance)
    var A_broadcasted = sc.broadcast(A)
    var B_broadcasted = sc.broadcast(B)
    var C_broadcasted = sc.broadcast(C)
    val T_broadcasted = sc.broadcast(T)
    var A_prev = DenseMatrix.zeros[Double](dimK, dimK)
    var lambda: breeze.linalg.DenseVector[Double] = DenseVector.zeros[Double](dimK)
    var mode: Int = 2
    var iter: Int = 0
    breakable {
      while (maxIterations <= 0 || iter < maxIterations) {
        mode = (mode + 1) % 3
        if (mode == 0) {
          iter = iter + 1
          if (isConverged(A_prev, A, tolerance)) {
            break
            // A_prev = A.copy
          }
          A_prev = A.copy
        }
        var A_array: Array[DenseVector[Double]] = new Array[DenseVector[Double]](dimK)
        A_array = sc.parallelize(0 until dimK, slices).map(i => updateALSiteration(i, dimK, A_broadcasted.value, C_broadcasted.value, B_broadcasted.value, T_broadcasted.value(i, ::).t)).collect()
        for (idx: Int <- 0 until dimK) {
          A(idx, ::) := A_array(idx).t
        }
        lambda = colWiseNorm2(A, tolerance)
        A = matrixNormalization(A, tolerance)
        A_broadcasted = sc.broadcast(A)


        var B_array: Array[DenseVector[Double]] = new Array[DenseVector[Double]](dimK)
        B_array = sc.parallelize(0 until dimK, slices).map(i => updateALSiteration(i, dimK, B_broadcasted.value, A_broadcasted.value, C_broadcasted.value, T_broadcasted.value(i, ::).t)).collect()
        for (idx: Int <- 0 until dimK) {
          B(idx, ::) := B_array(idx).t
        }
        B = matrixNormalization(B, tolerance)
        B_broadcasted = sc.broadcast(B)

        var C_array: Array[DenseVector[Double]] = new Array[DenseVector[Double]](dimK)
        C_array = sc.parallelize(0 until dimK, slices).map(i => updateALSiteration(i, dimK, C_broadcasted.value, B_broadcasted.value, A_broadcasted.value, T_broadcasted.value(i, ::).t)).collect()
        for (idx: Int <- 0 until dimK) {
          C(idx, ::) := C_array(idx).t
        }
        C = matrixNormalization(C, tolerance)
        C_broadcasted = sc.broadcast(C)

        iter += 1
      }
    }
    /*    println("result eigenvectors: ")
        println(A)
        println()
        println("result eigenvectors.t * eigenvectors: ")
        println(A.t*A)*/
    (A, lambda)
  }


  def updateALSiteration(i: Int, dimK: Int, A_old: DenseMatrix[Double], B_old: DenseMatrix[Double], C_old: DenseMatrix[Double], T: DenseVector[Double]): DenseVector[Double] = {
    val A_new: DenseVector[Double] = DenseVector.zeros[Double](dimK)
    val Inverted: DenseMatrix[Double] = to_invert(C_old, B_old)

    var result = new DenseVector[Double](dimK)
    assert(T.length == dimK * dimK)
    val rhs: DenseVector[Double] = Multip_KhatrioRao(T, C_old, B_old)
    result = Inverted * rhs
    result
  }

  def project(ii: Int, dimVocab: Int, dimK: Int, alpha0: Double,
              eigenValues: breeze.linalg.DenseVector[Double], eigenVectors: breeze.linalg.DenseMatrix[Double],
              Wc: breeze.linalg.SparseVector[Double], tolerance: Double): breeze.linalg.DenseVector[Double] = {
    var offset = 0
    var result = breeze.linalg.DenseVector.zeros[Double](dimK)
    while (offset < Wc.activeSize) {
      val token: Int = Wc.indexAt(offset)
      val count: Double = Wc.valueAt(offset)
      // val S_row = S(token,::)

      result += eigenVectors(token, ::).t.map(x => x * count)

      offset += 1
    }
    val whitenedData = result :/ (eigenValues.map(x => math.sqrt(x) + tolerance))
    return whitenedData
  }

  def showWarning() {
    System.err.println(
      """WARN: This is a naive implementation of tensor ALS.""".stripMargin)
  }

  def isConverged(oldA: DenseMatrix[Double], newA: DenseMatrix[Double], tolerance: Double): Boolean = {
    // val tolerance: Double = 0.0001
    if (oldA == null || oldA.size == 0) {
      return false
    }
    val numerator: Double = breeze.linalg.norm(newA.toDenseVector - oldA.toDenseVector)
    val denominator: Double = breeze.linalg.norm(newA.toDenseVector)
    val delta: Double = numerator / denominator
    return delta < tolerance
  }


  def gaussian(rows: Int, cols: Int, seed: Long, tolerance: Double): DenseMatrix[Double] = {
    val vectorizedOutputMatrix: Array[Double] = new Array[Double](rows * cols)
    val rand: scala.util.Random = new scala.util.Random(seed)
    // val rand: scala.util.Random = new scala.util.Random(System.currentTimeMillis)
    for (i: Int <- vectorizedOutputMatrix.indices) {
      vectorizedOutputMatrix(i) = rand.nextGaussian()
    }
    val gaussianMatrix: DenseMatrix[Double] = new DenseMatrix(rows, cols, vectorizedOutputMatrix)
    matrixNormalization(gaussianMatrix, tolerance)
  }

  def to_invert(c: DenseMatrix[Double], b: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ctc: DenseMatrix[Double] = c.t * c
    val btb: DenseMatrix[Double] = b.t * b
    val to_be_inverted: DenseMatrix[Double] = ctc :* btb
    return breeze.linalg.pinv(to_be_inverted)
  }


  def orthogonalizeMatCols(B: DenseMatrix[Double], tolerance: Double): DenseMatrix[Double] = {
    var A:DenseMatrix[Double] = B.copy

    for (j: Int <- 0 until A.cols) {
      for (i: Int <- 0 until j) {
        val dotij = A(::, j) dot A(::, i)
        A(::, j) :-= (A(::, i) :* dotij)

      }
      val normsq_sqrt: Double = Math.sqrt(A(::, j) dot A(::, j))
      var scale: Double = if (normsq_sqrt > tolerance) 1.0 / normsq_sqrt else 1e-12
      A(::, j) :*= scale
    }
    return A
  }


  def colWiseNorm2(A: breeze.linalg.DenseMatrix[Double], tolerance: Double): breeze.linalg.DenseVector[Double] = {
    // val tolerance : Double = 0.0001
    var normVec = breeze.linalg.DenseVector.zeros[Double](A.cols)
    for (i: Int <- 0 until A.cols) {
      val thisnorm: Double = Math.sqrt(A(::, i) dot A(::, i))
      normVec(i) = (if (thisnorm > tolerance) thisnorm else 1e-12)
    }
    normVec
  }


  def matrixNormalization(B: DenseMatrix[Double], tolerance: Double): DenseMatrix[Double] = {
    // val tolerance: Double = 0.0001
    var A: DenseMatrix[Double] = B.copy
    for (i: Int <- 0 until A.cols) {
      val thisnorm: Double = Math.sqrt(A(::, i) dot A(::, i))
      A(::, i) :*= (if (thisnorm > tolerance) (1.0 / thisnorm) else 1e-12)
    }
    return A
  }

  def KhatrioRao(A: DenseVector[Double], B: DenseVector[Double]): DenseVector[Double] = {
    var Out = DenseMatrix.zeros[Double](B.length, A.length)
    Out = B * A.t
    Out.flatten()
  }

  def KhatrioRao(A: DenseMatrix[Double], B: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(B.cols == A.cols)
    val Out = DenseMatrix.zeros[Double](B.rows * A.rows, A.cols)
    for (i: Int <- 0 until A.cols) {
      Out(::, i) := KhatrioRao(A(::, i), B(::, i))

    }
    Out
  }


  def Multip_KhatrioRao(T: DenseVector[Double], C: DenseVector[Double], B: DenseVector[Double]): Double = {
    var longvec = DenseVector.zeros[Double](C.length * B.length)
    longvec = KhatrioRao(C, B)
    T dot longvec
  }

  def Multip_KhatrioRao(T: DenseVector[Double], C: DenseMatrix[Double], B: DenseMatrix[Double]): DenseVector[Double] = {
    assert(B.cols == C.cols)
    val Out = DenseVector.zeros[Double](C.cols)
    for (i: Int <- 0 until C.cols) {
      Out(i) = Multip_KhatrioRao(T, C(::, i), B(::, i))

    }
    Out
  }


  def Multip_KhatrioRao(T: DenseMatrix[Double], C: DenseMatrix[Double], B: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(B.cols == C.cols)
    val Out = DenseMatrix.zeros[Double](C.cols, T.rows)
    for (i: Int <- 0 until T.rows) {
      var thisRowOfT = DenseVector.zeros[Double](T.cols)
      thisRowOfT = T(i, ::).t
      Out(::, i) := Multip_KhatrioRao(thisRowOfT, C, B)

    }
    Out.t
  }

  def processDocuments(sc: SparkContext, paths: Seq[String], stopwordFile: String, vocabSize: Int): (RDD[(Long, Double, breeze.linalg.SparseVector[Double])], Array[String], Int) = {
    val textRDD: RDD[String] = sc.textFile(paths.mkString(","))
    // Split text into words
    val tokenizer: SimpleTokenizer = new SimpleTokenizer(sc, stopwordFile)
    val tokenized: RDD[(Long, IndexedSeq[String])] = textRDD.zipWithIndex().map { case (text, id) =>
      id -> tokenizer.getWords(text)
    }
    tokenized.cache()

    // Counts words: RDD[(word, wordCount)]
    val wordCounts: RDD[(String, Long)] = tokenized
      .flatMap { case (_, tokens) => tokens.map(_ -> 1L) }
      .reduceByKey(_ + _)
    wordCounts.cache()
    val fullVocabSize: Long = wordCounts.count()

    // Select vocab
    //  (vocab: Map[word -> id], total tokens after selecting vocab)
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
      var values: Array[Double] = indices.map(i => wc(i).toDouble)
      val len: Double = values.sum
      // values = values.map(x => x/len)
      // val sb: Vector = Vectors.sparse(vocab.size, indices, values)
      var sb: breeze.linalg.SparseVector[Double] = {
        new breeze.linalg.SparseVector[Double](indices, values, vocab.size)
      }
      (id, len, sb)
    }
    val vocabarray: Array[String] = new Array[String](vocab.size)
    vocab.foreach { case (term, i) => vocabarray(i) = term }

    return (mydocuments, vocabarray, selectedTokenCount.toInt)
  }

  def processDocuments_synthetic(sc: SparkContext, paths: Seq[String], vocabSize: Int): (RDD[(Long, Double, SparseVector[Double])], Array[String], Int) ={
    val mypath: String = paths.mkString(",")
    println(mypath)
    val mylabeledpoints: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, mypath)
    val mydocuments: RDD[(Long, Double, breeze.linalg.SparseVector[Double])] = mylabeledpoints.map(f => (f.label.toLong, f.features.toArray.sum, new breeze.linalg.SparseVector[Double](f.features.toSparse.indices, f.features.toSparse.values, f.features.toSparse.size)))
    val mydocuments_collected: Array[(Long, Double, SparseVector[Double])] = mydocuments.collect()
    val vocabsize = mydocuments.collect()(0)._3.length
    val vocabarray: Array[String] = (0 until vocabsize).toArray.map(x => x.toString)
    (mydocuments, vocabarray, vocabsize)
  }

  def simplexProj(V: DenseVector[Double], tolerance: Double): DenseVector[Double]={
    //      val eps:Double = tolerance
    val z:Double = 1.0
    val len: Int = V.length
    var U: DenseVector[Double] = DenseVector(V.copy.toArray.sortWith(_ > _))
    var cums: DenseVector[Double] = DenseVector(Cumsum(U.toArray).map(x => x-1))
    val Index: DenseVector[Double] = DenseVector((1 to (len + 1)).toArray.map(x => 1.0/(x.toDouble)))
    val InterVec: DenseVector[Double] = cums :* Index
    val TobefindMax: DenseVector[Double] = U - InterVec
    var maxIndex : Long = 0
    // find maxIndex
    breakable{
      for (i: Int<- 0 until len){
        if (TobefindMax(len - i - 1) > 0){
          maxIndex = len - i - 1
          break
        }
      }
    }
    val theta: Double = InterVec(maxIndex.toInt)
    val W: DenseVector[Double] = V.map(x => x - theta)
    val P_norm: DenseVector[Double] = W.map(x => if (x > 0) x else 0)
    P_norm
  }
  def simplexProj_Matrix(M :DenseMatrix[Double], tolerance: Double): DenseMatrix[Double] ={
    var M_onSimplex: DenseMatrix[Double] = DenseMatrix.zeros[Double](M.rows, M.cols)
    for(i: Int<- 0 until M.cols){
      val thisColumn = M(::,i)

      val tmp1 = simplexProj(thisColumn, tolerance)
      val tmp2 = simplexProj(-thisColumn, tolerance)
      val err1:Double = breeze.linalg.norm(tmp1 - thisColumn)
      val err2:Double = breeze.linalg.norm(tmp2 - thisColumn)
      if(err1 > err2){
        M_onSimplex(::,i) := tmp2
      }
      else{
        M_onSimplex(::,i) := tmp1
      }
    }
    M_onSimplex
  }

}

// scalastyle:on println
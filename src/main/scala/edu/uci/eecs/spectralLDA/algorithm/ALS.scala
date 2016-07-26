 package edu.uci.eecs.spectralLDA.algorithm

 /**
 * Tensor Decomposition Algorithms.
 * Alternating Least Square algorithm is implemented.
 * Created by Furong Huang on 11/2/15.
 */
 import edu.uci.eecs.spectralLDA.utils.AlgebraUtil
 import edu.uci.eecs.spectralLDA.datamoments.DataCumulant
 import breeze.linalg.{DenseMatrix, DenseVector}
 import breeze.stats.distributions.{Rand, RandBasis, Gaussian}
 import org.apache.spark.rdd.RDD
 import org.apache.spark.SparkContext

 import scalaxy.loops._
 import scala.language.postfixOps
 import edu.uci.eecs.spectralLDA.utils.NonNegativeAdjustment

class ALS(dimK: Int, myData: DataCumulant) extends Serializable{

  def run(sc:SparkContext, maxIterations: Int)
         (implicit randBasis: RandBasis = Rand)
     : (DenseMatrix[Double], DenseVector[Double])={
    val T: breeze.linalg.DenseMatrix[Double] = myData.thirdOrderMoments
    val unwhiteningMatrix: DenseMatrix[Double] = myData.unwhiteningMatrix

    val gaussian = Gaussian(mu = 0.0, sigma = 1.0)
    var A: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)
    var B: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)
    var C: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)

    var A_prev = DenseMatrix.zeros[Double](dimK, dimK)
    var lambda: breeze.linalg.DenseVector[Double] = DenseVector.zeros[Double](dimK)

    println("Start ALS iterations...")
    var iter: Int = 0
    val T_RDD:RDD[DenseVector[Double]] = toRDD(sc,T)
    while ((iter == 0) || ((iter < maxIterations) && !AlgebraUtil.isConverged(A_prev, A))) {
      A_prev = A.copy

      // println("Mode A...")
      val A_array: Array[DenseVector[Double]] = T_RDD.map(thisT => updateALSiteration(dimK, A, C, B, thisT)).collect()
      for (idx <- 0 until dimK optimized) {
        A(idx, ::) := A_array(idx).t
      }
      lambda = AlgebraUtil.colWiseNorm2(A)
      A = AlgebraUtil.matrixNormalization(A)

      // println("Mode B...")
      val B_array: Array[DenseVector[Double]] = T_RDD.map(thisT => updateALSiteration(dimK, B, A, C,thisT)).collect()
      for (idx <- 0 until dimK optimized) {
        B(idx, ::) := B_array(idx).t
      }
      B = AlgebraUtil.matrixNormalization(B)

      // println("Mode C...")
      val C_array: Array[DenseVector[Double]] = T_RDD.map(thisT => updateALSiteration(dimK, C, B, A,thisT)).collect()
      for (idx <- 0 until dimK optimized) {
        C(idx, ::) := C_array(idx).t
      }
      C = AlgebraUtil.matrixNormalization(C)

      iter += 1
    }
    println("Finished ALS iterations.")

    val whitenedTopicWordMatrix: DenseMatrix[Double] = unwhiteningMatrix * A.copy
    val alpha: DenseVector[Double] = lambda.map(x => scala.math.pow(x, -2))
    val topicWordMatrix: breeze.linalg.DenseMatrix[Double] = whitenedTopicWordMatrix * breeze.linalg.diag(lambda)
    val topicWordMatrix_normed: DenseMatrix[Double] = NonNegativeAdjustment.simplexProj_Matrix(topicWordMatrix)
    (topicWordMatrix_normed, alpha)
  }

  private def toRDD(sc: SparkContext, m: DenseMatrix[Double]): RDD[DenseVector[Double]] = {
    val rows: Iterator[Array[Double]] = if (m.isTranspose) {
      m.data.grouped(m.majorStride)
    } else {
      val columns = m.data.grouped(m.rows)
      columns.toArray.transpose.iterator // Skip this if you want a column-major RDD.
    }
    val vectors = rows.map(row => new DenseVector[Double](row))
    sc.parallelize(vectors.to)
  }

  private def updateALSiteration(dimK: Int, A_old: DenseMatrix[Double], B_old: DenseMatrix[Double], C_old: DenseMatrix[Double], T: DenseVector[Double]): DenseVector[Double] = {
    val Inverted: DenseMatrix[Double] = AlgebraUtil.to_invert(C_old, B_old)

    assert(T.length == dimK * dimK)
    val rhs: DenseVector[Double] = AlgebraUtil.Multip_KhatrioRao(T, C_old, B_old)
    Inverted * rhs
  }
}

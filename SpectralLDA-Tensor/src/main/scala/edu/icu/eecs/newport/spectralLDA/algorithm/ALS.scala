 package edu.icu.eecs.newport.spectralLDA.algorithm

 /**
 * Tensor Decomposition Algorithms.
 * Alternating Least Square algorithm is implemented.
 * Created by Furong Huang on 11/2/15.
 */
 import edu.icu.eecs.newport.spectralLDA.utils.AlgebraUtil
 import edu.icu.eecs.newport.spectralLDA.datamoments.DataCumulant
 import breeze.linalg.{DenseMatrix, DenseVector}
 import org.apache.spark.rdd.RDD
 import org.apache.spark.SparkContext
 import scalaxy.loops._
 import scala.language.postfixOps
 import scala.util.control.Breaks._

class ALS(slices: Int, dimK: Int, myData: DataCumulant) extends Serializable{
  def run(sc:SparkContext, maxIterations: Int): (DenseMatrix[Double], DenseVector[Double])={
    val T: breeze.linalg.DenseMatrix[Double] = myData.thirdOrderMoments
    val unwhiteningMatrix: DenseMatrix[Double] = myData.unwhiteningMatrix
    val SEED_A: Long = System.currentTimeMillis
    val SEED_B: Long = System.currentTimeMillis
    val SEED_C: Long = System.currentTimeMillis
    var A: DenseMatrix[Double] = AlgebraUtil.gaussian(dimK, dimK, SEED_A)
    var B: DenseMatrix[Double] = AlgebraUtil.gaussian(dimK, dimK, SEED_B)
    var C: DenseMatrix[Double] = AlgebraUtil.gaussian(dimK, dimK, SEED_C)
//    var A_broadcasted = sc.broadcast(A)
//    var B_broadcasted = sc.broadcast(B)
//    var C_broadcasted = sc.broadcast(C)
//    val T_broadcasted = sc.broadcast(T)
    var A_prev = DenseMatrix.zeros[Double](dimK, dimK)
    var lambda: breeze.linalg.DenseVector[Double] = DenseVector.zeros[Double](dimK)
    var mode: Int = 2
    var iter: Int = 0
    println("Pseudo RDD...")
    val pseudoRDD = sc.parallelize(0 until dimK, slices)
    println("Start ALS iterations...")
    breakable {
      while (maxIterations <= 0 || iter < maxIterations) {
        mode = (mode + 1) % 3
        if (mode == 0) {
          iter = iter + 1
          if (AlgebraUtil.isConverged(A_prev, A)) {
            break()
          }
          A_prev = A.copy
        }

        val T_RDD:RDD[DenseVector[Double]] = toRDD(sc,T)

        // println("Mode A...")
        val A_array: Array[DenseVector[Double]] = T_RDD.map(thisT => updateALSiteration(dimK, A, C, B, thisT)).collect()
        // A_array = pseudoRDD.map(i => updateALSiteration(dimK, A, C, B, T(i, ::).t)).collect()
        for (idx <- 0 until dimK optimized) {
          A(idx, ::) := A_array(idx).t
        }
        lambda = AlgebraUtil.colWiseNorm2(A)
        A = AlgebraUtil.matrixNormalization(A)
        // A_broadcasted = sc.broadcast(A)


        // println("Mode B...")
        val B_array: Array[DenseVector[Double]] = T_RDD.map(thisT => updateALSiteration(dimK, B, A, C,thisT)).collect()
        // B_array = pseudoRDD.map(i => updateALSiteration(dimK, B, A, C, T(i, ::).t)).collect()
        for (idx <- 0 until dimK optimized) {
          B(idx, ::) := B_array(idx).t
        }
        B = AlgebraUtil.matrixNormalization(B)
        // B_broadcasted = sc.broadcast(B)

        // println("Mode C...")
        val C_array: Array[DenseVector[Double]] = T_RDD.map(thisT => updateALSiteration(dimK, C, B, A,thisT)).collect()
        // C_array = pseudoRDD.map(i => updateALSiteration(dimK, C, B, A, T(i, ::).t)).collect()
        for (idx <- 0 until dimK optimized) {
          C(idx, ::) := C_array(idx).t
        }
        C = AlgebraUtil.matrixNormalization(C)
        // C_broadcasted = sc.broadcast(C)

        iter += 1
      }
    }
    println("Finished ALS iterations.")

    val whitenedTopicWordMatrix: DenseMatrix[Double] = unwhiteningMatrix * A.copy
    val alpha: DenseVector[Double] = lambda.map(x => scala.math.pow(x, -2))
    val topicWordMatrix: breeze.linalg.DenseMatrix[Double] = whitenedTopicWordMatrix * breeze.linalg.diag(lambda)
    val topicWordMatrix_normed: breeze.linalg.DenseMatrix[Double] = simplexProj_Matrix(topicWordMatrix)
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

  private def simplexProj_Matrix(M :DenseMatrix[Double]): DenseMatrix[Double] ={
    val M_onSimplex: DenseMatrix[Double] = DenseMatrix.zeros[Double](M.rows, M.cols)
    for(i <- 0 until M.cols optimized){
      val thisColumn = M(::,i)

      val tmp1 = simplexProj(thisColumn)
      val tmp2 = simplexProj(-thisColumn)
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

  private def simplexProj(V: DenseVector[Double]): DenseVector[Double]={
    // val z:Double = 1.0
    val len: Int = V.length
    val U: DenseVector[Double] = DenseVector(V.copy.toArray.sortWith(_ > _))
    val cums: DenseVector[Double] = DenseVector(AlgebraUtil.Cumsum(U.toArray).map(x => x-1))
    val Index: DenseVector[Double] = DenseVector((1 to (len + 1)).toArray.map(x => 1.0/x.toDouble))
    val InterVec: DenseVector[Double] = cums :* Index
    val TobefindMax: DenseVector[Double] = U - InterVec
    var maxIndex : Int = 0
    // find maxIndex
    breakable{
      for (i <- 0 until len optimized){
        if (TobefindMax(len - i - 1) > 0){
          maxIndex = len - i - 1
          break()
        }
      }
    }
    val theta: Double = InterVec(maxIndex)
    val W: DenseVector[Double] = V.map(x => x - theta)
    val P_norm: DenseVector[Double] = W.map(x => if (x > 0) x else 0)
    P_norm
  }

}

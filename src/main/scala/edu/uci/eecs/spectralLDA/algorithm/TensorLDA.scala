package edu.uci.eecs.spectralLDA.algorithm

/**
 * Tensor Decomposition Algorithms.
 * Alternating Least Square algorithm is implemented.
 * Created by Furong Huang on 11/2/15.
 */
import edu.uci.eecs.spectralLDA.datamoments.DataCumulant
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, sum}
import breeze.stats.distributions.{Rand, RandBasis}
import org.apache.spark.rdd.RDD

class TensorLDA(dimK: Int,
                alpha0: Double,
                maxIterations: Int = 200,
                tolerance: Double = 1e-9)
                extends Serializable {

  def fit(documents: RDD[(Long, SparseVector[Double])])
         (implicit randBasis: RandBasis = Rand)
          : (DenseMatrix[Double], DenseVector[Double]) = {
    val documents_ = documents map {
      case (id, wc) => (id, sum(wc), wc)
    }

    val myData: DataCumulant = DataCumulant.getDataCumulant(
      dimK, alpha0,
      tolerance,
      documents_
    )

    val myALS: ALS = new ALS(dimK, myData)
    myALS.run(documents.sparkContext, maxIterations)
  }

}

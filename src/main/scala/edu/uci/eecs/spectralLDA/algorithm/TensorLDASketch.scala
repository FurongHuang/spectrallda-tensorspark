package edu.uci.eecs.spectralLDA.algorithm

/**
  * Tensor Decomposition Algorithms.
  * Alternating Least Square algorithm is implemented.
  */
import edu.uci.eecs.spectralLDA.datamoments.{DataCumulant, DataCumulantSketch}
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, sum}
import breeze.stats.distributions.{Rand, RandBasis}
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher
import org.apache.spark.rdd.RDD

class TensorLDASketch(dimK: Int,
                      alpha0: Double,
                      maxIterations: Int = 1000,
                      sketcher: TensorSketcher[Double, Double],
                      randomisedSVD: Boolean = true,
                      nonNegativeDocumentConcentration: Boolean = true)
                     (implicit tolerance: Double = 1e-9)
  extends Serializable {

  def fit(documents: RDD[(Long, SparseVector[Double])])
         (implicit randBasis: RandBasis = Rand)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    val myDataSketch: DataCumulantSketch = DataCumulantSketch.getDataCumulant(
      dimK, alpha0,
      documents,
      sketcher,
      randomisedSVD = randomisedSVD
    )

    val myALS: ALSSketch = new ALSSketch(
      dimK,
      myDataSketch,
      sketcher,
      nonNegativeDocumentConcentration = nonNegativeDocumentConcentration
    )

    val (beta, alpha) = myALS.run
    (beta, alpha * alpha0)
  }

}
package edu.uci.eecs.spectralLDA.algorithm

/**
  * Tensor Decomposition Algorithms.
  * Alternating Least Square algorithm is implemented.
  */
import edu.uci.eecs.spectralLDA.datamoments.DataCumulantSketch
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, diag}
import breeze.stats.distributions.{Rand, RandBasis}
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher
import edu.uci.eecs.spectralLDA.utils.NonNegativeAdjustment
import org.apache.spark.rdd.RDD

class TensorLDASketch(dimK: Int,
                      alpha0: Double,
                      maxIterations: Int = 200,
                      sketcher: TensorSketcher[Double, Double],
                      randomisedSVD: Boolean = true,
                      nonNegativeDocumentConcentration: Boolean = true)
                     (implicit tolerance: Double = 1e-9)
  extends Serializable {
  assert(sketcher.n forall { _ == dimK }, s"The sketcher must work on symmetric tensors of shape ($dimK, ..., $dimK).")

  def fit(documents: RDD[(Long, SparseVector[Double])])
         (implicit randBasis: RandBasis = Rand)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    val cumulantSketch: DataCumulantSketch = DataCumulantSketch.getDataCumulant(
      dimK, alpha0,
      documents,
      sketcher,
      randomisedSVD = randomisedSVD
    )

    val myALS: ALSSketch = new ALSSketch(
      dimK,
      cumulantSketch.fftSketchWhitenedM3,
      sketcher,
      maxIterations = maxIterations
    )

    val (nu: DenseMatrix[Double], lambda: DenseVector[Double]) = myALS.run

    // unwhiten the results
    val alpha: DenseVector[Double] = lambda.map(x => scala.math.pow(x, -2)) * alpha0
    val topicWordMatrix: breeze.linalg.DenseMatrix[Double] = cumulantSketch.unwhiteningMatrix * nu * diag(lambda)

    // non-negativity adjustment for the word distributions per topic
    if (nonNegativeDocumentConcentration) {
      val topicWordMatrix_normed = NonNegativeAdjustment.simplexProj_Matrix(topicWordMatrix)
      (topicWordMatrix_normed, alpha)
    }
    else {
      (topicWordMatrix, alpha)
    }
  }
}
package edu.uci.eecs.spectralLDA.algorithm

/**
  * Tensor Decomposition Algorithms.
  * Alternating Least Square algorithm is implemented.
  */
import edu.uci.eecs.spectralLDA.datamoments.DataCumulantSketch
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, diag, max, min}
import breeze.numerics._
import breeze.stats.distributions.{Rand, RandBasis}
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher
import edu.uci.eecs.spectralLDA.utils.NonNegativeAdjustment
import org.apache.spark.rdd.RDD

class TensorLDASketch(dimK: Int,
                      alpha0: Double,
                      maxIterations: Int = 200,
                      sketcher: TensorSketcher[Double, Double],
                      m2ConditionNumberUB: Double = Double.PositiveInfinity,
                      randomisedSVD: Boolean = true,
                      nonNegativeDocumentConcentration: Boolean = true)
                     (implicit tolerance: Double = 1e-9)
  extends Serializable {
  assert(dimK > 0, "The number of topics dimK must be positive.")
  assert(alpha0 > 0, "The topic concentration alpha0 must be positive.")
  assert(maxIterations > 0, "The number of iterations for ALS must be positive.")
  assert(sketcher.n forall { _ == dimK }, s"The sketcher must work on symmetric tensors of shape ($dimK, ..., $dimK).")

  def fit(documents: RDD[(Long, SparseVector[Double])])
         (implicit randBasis: RandBasis = Rand)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    val cumulantSketch: DataCumulantSketch = DataCumulantSketch.getDataCumulant(
      dimK, alpha0,
      documents,
      sketcher,
      m2ConditionNumberUB = m2ConditionNumberUB,
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
    // unwhitening matrix: $(W^T)^{-1}=U\Sigma^{1/2}$
    val unwhiteningMatrix = cumulantSketch.eigenVectorsM2 * diag(sqrt(cumulantSketch.eigenValuesM2))

    val alpha: DenseVector[Double] = lambda.map(x => scala.math.pow(x, -2)) * alpha0
    val topicWordMatrix: breeze.linalg.DenseMatrix[Double] = unwhiteningMatrix * nu * diag(lambda)

    // Diagnostic information: the ratio of the maximum to the minimum of the
    // top k eigenvalues of shifted M2
    //
    // If it's too large (>10), the algorithm may not be able to output reasonable results.
    // It could be due to some very frequent (low IDF) words or that we specified dimK
    // larger than the rank of the shifted M2.
    val m2ConditionNumber = max(cumulantSketch.eigenValuesM2) / min(cumulantSketch.eigenValuesM2)
    println(s"Shifted M2 top $dimK eigenvalues: ${cumulantSketch.eigenValuesM2}")
    println(s"Shifted M2 condition number: $m2ConditionNumber")
    println("If the condition number is too large (e.g. >10), the algorithm may not be able to " +
            "output reasonable results. It could be due to the existence of very frequent words " +
            "across the documents or that the specified k is larger than the true number of topics.")

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
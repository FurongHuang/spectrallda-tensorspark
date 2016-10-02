package edu.uci.eecs.spectralLDA.algorithm

/**
* Tensor Decomposition Algorithms.
* Alternating Least Square algorithm is implemented.
* Created by Furong Huang on 11/2/15.
*/

import edu.uci.eecs.spectralLDA.utils.{AlgebraUtil, TensorOps}
import breeze.linalg.{*, DenseMatrix, DenseVector, diag, inv, max, min, norm, qr}
import breeze.stats.distributions.{Gaussian, Rand, RandBasis}

/** Tensor decomposition by Alternating Least Square (ALS)
  *
  * Suppose dimK-by-dimK-by-dimK symmetric tensor T can be decomposed as sum of rank-1 tensors
  *
  * $$ T = \sum_{i=1}^{dimK} \lambda_i a_i\otimes b_i\otimes c_i $$
  *
  * If we pool all \lambda_i in the vector \Lambda, all the column vectors \lambda_i a_i in A,
  * b_i in B, c_i in C, then
  *
  * $$ T^{1} = A \diag(\Lambda)(C \khatri-rao product B)^{\top} $$
  *
  * where T^{1} is a dimK-by-(dimK^2) matrix for the unfolded T.
  *
  * @param dimK               tensor T is of shape dimK-by-dimK-by-dimK
  * @param thirdOrderMoments  dimK-by-(dimK*dimK) matrix for the unfolded 3rd-order moments
  *                           $\sum_{i=1}^k\alpha_i\beta_i^{\otimes 3}$
  * @param maxIterations      max iterations for the ALS algorithm
  */
class ALS(dimK: Int,
          thirdOrderMoments: DenseMatrix[Double],
          maxIterations: Int = 200)
  extends Serializable {
  assert(dimK > 0, "The number of topics dimK must be positive.")
  assert(thirdOrderMoments.rows == dimK && thirdOrderMoments.cols == dimK * dimK,
    "The thirdOrderMoments must be dimK-by-(dimK * dimK) unfolded matrix")

  /** Run Alternating Least Squares (ALS)
    *
    * Compute the best approximating rank-$k$ tensor $\sum_{i=1}^k\alpha_i\beta_i^{\otimes 3}$
    *
    * @param randBasis   default random seed
    * @return            three dimK-by-dimK matrices with all the $beta_i$ as columns,
    *                    length-dimK vector for all the eigenvalues
    */
  def run(implicit randBasis: RandBasis = Rand, restarts: Int = 5)
     : (DenseMatrix[Double], DenseMatrix[Double],
        DenseMatrix[Double], DenseVector[Double])={
    assert(restarts > 0, "Number of restarts for ALS must be positive.")

    val gaussian = Gaussian(mu = 0.0, sigma = 1.0)

    var optimalA = DenseMatrix.zeros[Double](dimK, dimK)
    var optimalB = DenseMatrix.zeros[Double](dimK, dimK)
    var optimalC = DenseMatrix.zeros[Double](dimK, dimK)
    var optimalLambda = DenseVector.zeros[Double](dimK)

    var reconstructedLoss: Double = 0.0
    var optimalReconstructedLoss: Double = Double.PositiveInfinity

    for (s <- 0 until restarts) {
      val qr.QR(a0, _) = qr(DenseMatrix.rand[Double](dimK, dimK, gaussian))
      val qr.QR(b0, _) = qr(DenseMatrix.rand[Double](dimK, dimK, gaussian))
      val qr.QR(c0, _) = qr(DenseMatrix.rand[Double](dimK, dimK, gaussian))

      var A = a0
      var B = b0
      var C = c0

      var A_prev = DenseMatrix.zeros[Double](dimK, dimK)
      var lambda: breeze.linalg.DenseVector[Double] = DenseVector.zeros[Double](dimK)

      println("Start ALS iterations...")
      var iter: Int = 0
      while ((iter == 0) ||
        ((iter < maxIterations) && !AlgebraUtil.isConverged(A_prev, A)(dotThreshold = 0.999))) {
        A_prev = A.copy

        val (updatedA, updatedLambda1) = updateOrthoALSIteration3(thirdOrderMoments, B, C)
        A = updatedA
        lambda = updatedLambda1

        val (updatedB, updatedLambda2) = updateOrthoALSIteration3(thirdOrderMoments, C, A)
        B = updatedB
        lambda = updatedLambda2

        val (updatedC, updatedLambda3) = updateOrthoALSIteration3(thirdOrderMoments, A, B)
        C = updatedC
        lambda = updatedLambda3

        println(s"iter $iter\tlambda: max ${max(lambda)}, min ${min(lambda)}")

        iter += 1
      }
      println("Finished ALS iterations.")

      reconstructedLoss = TensorOps.dmatrixNorm(thirdOrderMoments - A * diag(lambda) * TensorOps.krprod(C, B).t)
      println(s"Reconstructed loss: $reconstructedLoss\tOptimal reconstructed loss: $optimalReconstructedLoss")

      if (reconstructedLoss < optimalReconstructedLoss) {
        optimalA = A
        optimalB = B
        optimalC = C
        optimalLambda = lambda
        optimalReconstructedLoss = reconstructedLoss
      }
    }

    (optimalA, optimalB, optimalC, optimalLambda)
  }

  private def updateALSIteration(unfoldedM3: DenseMatrix[Double],
                                 B: DenseMatrix[Double],
                                 C: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
    val updatedA = unfoldedM3 * TensorOps.krprod(C, B) * TensorOps.to_invert(C, B)
    val lambda = norm(updatedA(::, *)).toDenseVector
    (AlgebraUtil.matrixNormalization(updatedA), lambda)
  }

  private def updateOrthoALSIteration1(unfoldedM3: DenseMatrix[Double],
                                       B: DenseMatrix[Double],
                                       C: DenseMatrix[Double])
                                      (implicit nonOrthoPenalty: Double = 10.0)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    val (updatedA, lambda) = updateALSIteration(unfoldedM3, B, C)
    val qr.QR(q, _) = qr(updatedA)
    (q, lambda)
  }

  private def updateOrthoALSIteration2(unfoldedM3: DenseMatrix[Double],
                                       B: DenseMatrix[Double],
                                       C: DenseMatrix[Double])
  : (DenseMatrix[Double], DenseVector[Double]) = {
    val updatedA = unfoldedM3 * TensorOps.krprod(C, B) * TensorOps.to_invert(C, B)
    val qr.QR(q, r) = qr(updatedA)
    (q, diag(r))
  }

  private def updateOrthoALSIteration3(unfoldedM3: DenseMatrix[Double],
                                       B: DenseMatrix[Double],
                                       C: DenseMatrix[Double])
                                      (implicit
                                       penalty: Double = 10.0,
                                       step: Double = 1e-3,
                                       maxIter: Int = 100,
                                       tol: Double = 1e-4)
  : (DenseMatrix[Double], DenseVector[Double]) = {
    val (updatedA, lambda) = updateALSIteration(unfoldedM3, B, C)
    var orthoA = updatedA.copy
    var nextOrthoA = updatedA.copy

    var i = 0
    val eyeK = DenseMatrix.eye[Double](dimK)

    if (TensorOps.dmatrixNorm(updatedA.t * updatedA - eyeK) / dimK.toDouble > 1e-4) {
      while ((i == 0) || (i < maxIter &&
        TensorOps.dmatrixNorm(nextOrthoA - orthoA) > tol * TensorOps.dmatrixNorm(orthoA))) {
        orthoA = nextOrthoA

        val h = eyeK + penalty * (orthoA.t * orthoA - eyeK)
        val grad = orthoA * h - updatedA

        nextOrthoA = orthoA - step * grad

        i += 1
      }
    }

    val normPrior = TensorOps.dmatrixNorm(updatedA.t * updatedA - eyeK)
    val normPost = TensorOps.dmatrixNorm(nextOrthoA.t * nextOrthoA - eyeK)
    println(s"norm(A^T A-I) prior proximal op: ${normPrior}\tpost proximal op: ${normPost}\tProximal steps: $i")
    (nextOrthoA, lambda)
  }
}

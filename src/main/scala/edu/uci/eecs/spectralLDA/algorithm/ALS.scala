package edu.uci.eecs.spectralLDA.algorithm

/**
* Tensor Decomposition Algorithms.
* Alternating Least Square algorithm is implemented.
* Created by Furong Huang on 11/2/15.
*/

import edu.uci.eecs.spectralLDA.utils.{AlgebraUtil, TensorOps}
import breeze.linalg.{*, DenseMatrix, DenseVector, norm, max, min}
import breeze.stats.distributions.{Gaussian, Rand, RandBasis}

class ALS(dimK: Int,
          thirdOrderMoments: DenseMatrix[Double],
          maxIterations: Int = 200)
  extends Serializable {

  def run(implicit randBasis: RandBasis = Rand)
     : (DenseMatrix[Double], DenseVector[Double])={
    val gaussian = Gaussian(mu = 0.0, sigma = 1.0)
    var A: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)
    var B: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)
    var C: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)

    var A_prev = DenseMatrix.zeros[Double](dimK, dimK)
    var lambda: breeze.linalg.DenseVector[Double] = DenseVector.zeros[Double](dimK)

    println("Start ALS iterations...")
    var iter: Int = 0
    while ((iter == 0) || ((iter < maxIterations) && !AlgebraUtil.isConverged(A_prev, A))) {
      A_prev = A.copy

      // println("Mode A...")
      A = updateALSIteration(thirdOrderMoments, B, C)
      lambda = norm(A(::, *)).toDenseVector
      println(s"iter $iter\tlambda: max ${max(lambda)}, min ${min(lambda)}")
      A = AlgebraUtil.matrixNormalization(A)

      // println("Mode B...")
      B = updateALSIteration(thirdOrderMoments, C, A)
      B = AlgebraUtil.matrixNormalization(B)

      // println("Mode C...")
      C = updateALSIteration(thirdOrderMoments, A, B)
      C = AlgebraUtil.matrixNormalization(C)

      iter += 1
    }
    println("Finished ALS iterations.")

    (A, lambda)
  }

  private def updateALSIteration(thirdOrderMoments: DenseMatrix[Double],
                                 B: DenseMatrix[Double],
                                 C: DenseMatrix[Double]): DenseMatrix[Double] = {
    thirdOrderMoments * TensorOps.krprod(C, B) * TensorOps.to_invert(C, B)
  }
}

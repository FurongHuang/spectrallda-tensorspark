package edu.uci.eecs.spectralLDA.algorithm

/**
  * Tensor Decomposition Algorithms.
  * Alternating Least Square algorithm is implemented.
  */
import edu.uci.eecs.spectralLDA.utils.{AlgebraUtil, NonNegativeAdjustment}
import breeze.linalg.{*, DenseMatrix, DenseVector, diag, max, min, norm, pinv, sum}
import breeze.signal.{fourierTr, iFourierTr}
import breeze.math.Complex
import breeze.numerics._
import breeze.stats.distributions.{Gaussian, Rand, RandBasis}
import breeze.stats.median
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher

import scala.language.postfixOps

/** Sketched tensor decomposition by Alternating Least Square (ALS)
  *
  * Suppose dimK-by-dimK-by-dimK tensor T can be decomposed as sum of rank-1 tensors
  *
  *    T = \sum_{i=1}^{dimK} \lambda_i a_i\otimes b_i\otimes c_i
  *
  * If we pool all the column vectors \lambda_i a_i in A, b_i in B, c_i in C, then
  *
  *    T^{1} = A (C \khatri-rao product B)^{\top}
  *
  * where T^{1} is a dimK-by-(dimK^2) matrix for the unfolded T.
  *
  * @param dimK          tensor T is of shape dimK-by-dimK-by-dimK
  * @param fft_sketch_T  FFT of sketched tensor T
  * @param sketcher      the sketching facility
  * @param maxIterations max iterations for the ALS algorithm
  */
class ALSSketch(dimK: Int,
                fft_sketch_T: DenseMatrix[Complex],
                sketcher: TensorSketcher[Double, Double],
                maxIterations: Int = 200
                ) extends Serializable {

  def run(implicit randBasis: RandBasis = Rand)
        : (DenseMatrix[Double], DenseVector[Double]) = {
    val gaussian = Gaussian(mu = 0.0, sigma = 1.0)
    var A: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)
    var B: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)
    var C: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)

    var A_prev = DenseMatrix.zeros[Double](dimK, dimK)
    var lambda: breeze.linalg.DenseVector[Double] = DenseVector.zeros[Double](dimK)
    var iter: Int = 0

    println("Start ALS iterations...")

    while ((iter == 0) || ((iter < maxIterations) && !AlgebraUtil.isConverged(A_prev, A))) {
      A_prev = A.copy

      // println("Mode A...")
      A = updateALSiteration(fft_sketch_T, B, C, sketcher)
      lambda = norm(A(::, *)).toDenseVector
      println(s"iter $iter\tlambda: max ${max(lambda)}, min ${min(lambda)}")
      A = AlgebraUtil.matrixNormalization(A)

      // println("Mode B...")
      B = updateALSiteration(fft_sketch_T, C, A, sketcher)
      B = AlgebraUtil.matrixNormalization(B)

      // println("Mode C...")
      C = updateALSiteration(fft_sketch_T, A, B, sketcher)
      C = AlgebraUtil.matrixNormalization(C)

      iter += 1
    }
    println("Finished ALS iterations.")

    (A, lambda)
  }

  def updateALSiteration(fft_sketch_T: DenseMatrix[Complex],
                                 B: DenseMatrix[Double],
                                 C: DenseMatrix[Double],
                                 sketcher: TensorSketcher[Double, Double])
          : DenseMatrix[Double] = {
    // pinv((C^T C) :* (B^T B))
    val Inverted: DenseMatrix[Double] = AlgebraUtil.to_invert(C, B)

    // T(C katri-rao dot B)
    val TIBC: DenseMatrix[Double] = TensorSketchOps.TIUV(fft_sketch_T, B, C, sketcher)

    // T * (C katri-rao dot B) * pinv((C^T C) :* (B^T B))
    // i.e T * pinv((C katri-rao dot B)^T)
    TIBC * Inverted
  }
}

/** Non-negative sketched tensor decomposition by Alternating Least Square (ALS)
  *
  * Suppose dimK-by-dimK-by-dimK tensor T can be decomposed as sum of rank-1 tensors
  *
  *    T = \sum_{i=1}^{dimK} \lambda_i a_i\otimes b_i\otimes c_i
  *
  * If we pool all the column vectors \lambda_i a_i in A, b_i in B, c_i in C, then
  *
  *    T^{1} = A (C \khatri-rao product B)^{\top}
  *
  * where T^{1} is a dimK-by-(dimK^2) matrix for the unfolded T.
  *
  * Furthermore, given a V-by-dimK unwhitening matrix H, the solution will satisfy
  *
  *    Ha_i >= 0, Hb_i >= 0, Hc_i >= 0, and Ha_i, Hb_i, Hc_i sum up to 1, \forall i\in [1,dimK].
  *
  * @param dimK            tensor T is of shape dimK-by-dimK-by-dimK
  * @param fft_sketch_T    FFT of sketched tensor T
  * @param sketcher        the sketching facility
  * @param eigenVectorsM2  eigenvectors of M2
  * @param eigenValuesM2   eigenvalues of M2
  * @param maxIterations   max iterations for the ALS algorithm
  */
class NNALSSketch(dimK: Int,
                  fft_sketch_T: DenseMatrix[Complex],
                  sketcher: TensorSketcher[Double, Double],
                  eigenVectorsM2: DenseMatrix[Double],
                  eigenValuesM2: DenseVector[Double],
                  maxIterations: Int = 200
                 )
  extends ALSSketch(dimK, fft_sketch_T, sketcher, maxIterations) {

  override def run(implicit randBasis: RandBasis)
      : (DenseMatrix[Double], DenseVector[Double]) = {
    val h = eigenVectorsM2 * diag(sqrt(eigenValuesM2))
    val hInv = diag(1.0 / sqrt(eigenValuesM2)) * eigenVectorsM2.t

    val gaussian = Gaussian(mu = 0.0, sigma = 1.0)
    var A: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)
    var B: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)
    var C: DenseMatrix[Double] = DenseMatrix.rand[Double](dimK, dimK, gaussian)

    var A_prev = DenseMatrix.zeros[Double](dimK, dimK)
    var lambda: breeze.linalg.DenseVector[Double] = DenseVector.zeros[Double](dimK)
    var iter: Int = 0

    println("Start ALS iterations...")

    while ((iter == 0) || ((iter < maxIterations) && !AlgebraUtil.isConverged(A_prev, A))) {
      A_prev = A.copy

      // println("Mode A...")
      val refA = updateALSiteration(fft_sketch_T, B, C, sketcher)
      A = hInv * NonNegativeAdjustment.simplexProj_Matrix(h * refA)
      println(s"iter $iter\tsum(HA) ${sum(h * A)}\tmin(HA) ${min(h * A)}")
      lambda = norm(A(::, *)).toDenseVector
      println(s"iter $iter\tlambda: max ${max(lambda)}, min ${min(lambda)}")
      A = AlgebraUtil.matrixNormalization(A)

      // println("Mode B...")
      val refB = updateALSiteration(fft_sketch_T, C, A, sketcher)
      B = hInv * NonNegativeAdjustment.simplexProj_Matrix(h * refB)
      B = AlgebraUtil.matrixNormalization(B)

      // println("Mode C...")
      val refC = updateALSiteration(fft_sketch_T, A, B, sketcher)
      C = hInv * NonNegativeAdjustment.simplexProj_Matrix(h * refC)
      C = AlgebraUtil.matrixNormalization(C)

      iter += 1
    }
    println("Finished ALS iterations.")

    (A, lambda)
  }
}

private[algorithm] object TensorSketchOps {
  /** Compute T(I, u, v) given the FFT of sketch_T
    *
    * T is an n-by-n-by-n tensor, for the orthogonalised M3
    *
    * @param fft_sketch_T FFT of sketch_T, with B rows where B is the number of hash families
    * @param u length-n vector
    * @param v length-n vector
    * @param sketcher count sketcher on n-by-n-by-n tensors
    * @return length-n vector for T(I, u, v)
    */
  private def TIuv(fft_sketch_T: DenseMatrix[Complex],
                   u: DenseVector[Double],
                   v: DenseVector[Double],
                   sketcher: TensorSketcher[Double, Double])
      : DenseVector[Double] = {
    val n = u.length

    val sketch_u: DenseMatrix[Double] = sketcher.sketch(u, 1)
    val sketch_v: DenseMatrix[Double] = sketcher.sketch(v, 2)

    val fft_sketch_u: DenseMatrix[Complex] = fourierTr(sketch_u(*, ::))
    val fft_sketch_v: DenseMatrix[Complex] = fourierTr(sketch_v(*, ::))

    val prod_fft: DenseMatrix[Complex] = (fft_sketch_T :* (fft_sketch_u map { _.conjugate })
      :* (fft_sketch_v map { _.conjugate }))
    val TIuv_lhs: DenseMatrix[Double] = iFourierTr(prod_fft(*, ::)) map { _.re }

    val all_inner_prod: DenseMatrix[Double] = DenseMatrix.zeros[Double](sketcher.B, n)
    for (hashFamilyId <- 0 until sketcher.B; i <- 0 until n) {
      all_inner_prod(hashFamilyId, i) = (TIuv_lhs(hashFamilyId, sketcher.h((hashFamilyId, 0, i)) % sketcher.b)
        * sketcher.xi((hashFamilyId, 0, i)))
    }

    val result = for {
        i <- 0 until n
    } yield median(all_inner_prod(::, i))

    DenseVector(result: _*)
  }

  /** Compute the matrix {T(I, U_i, V_i), 1\le i\le k} given sketch_T
    *
    * T is an n-by-n-by-n tensor, for the orthogonalised M3
    *
    * @param fft_sketch_T FFT of the sketch of T, with B rows where B is the number of hash families
    * @param U n-by-k matrix
    * @param V n-by-k matrix
    * @param sketcher count sketcher on n-by-n-by-n tensors
    * @return n-by-k matrix for {T(I, U_i, V_i), 1\le i\le k}
    */
  def TIUV(fft_sketch_T: DenseMatrix[Complex],
           U: DenseMatrix[Double],
           V: DenseMatrix[Double],
           sketcher: TensorSketcher[Double, Double])
      : DenseMatrix[Double] = {
    assert((U.rows == V.rows) && (U.cols == V.cols))
    assert(sketcher.n(0) == sketcher.n(1) && sketcher.n(1) == sketcher.n(2)
      && sketcher.n(0) == U.rows)

    val result: DenseMatrix[Double] = DenseMatrix.zeros[Double](U.rows, U.cols)
    for (j <- 0 until U.cols) {
      result(::, j) := TIuv(fft_sketch_T, U(::, j), V(::, j), sketcher)
    }

    result
  }
}

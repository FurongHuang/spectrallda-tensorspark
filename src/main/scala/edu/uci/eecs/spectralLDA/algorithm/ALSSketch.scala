package edu.uci.eecs.spectralLDA.algorithm

/**
  * Tensor Decomposition Algorithms.
  * Alternating Least Square algorithm is implemented.
  */
import edu.uci.eecs.spectralLDA.utils.AlgebraUtil
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.signal.{fourierTr, iFourierTr}
import breeze.math.Complex
import breeze.stats.distributions.{Rand, RandBasis}
import breeze.stats.median
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher

import scala.language.postfixOps

class ALSSketch(dimK: Int,
                fft_sketch_T: DenseMatrix[Complex],
                sketcher: TensorSketcher[Double, Double],
                maxIterations: Int = 1000
                ) extends Serializable {

  def run(implicit randBasis: RandBasis = Rand)
        : (DenseMatrix[Double], DenseVector[Double]) = {
    var A: DenseMatrix[Double] = AlgebraUtil.gaussian(dimK, dimK)
    var B: DenseMatrix[Double] = AlgebraUtil.gaussian(dimK, dimK)
    var C: DenseMatrix[Double] = AlgebraUtil.gaussian(dimK, dimK)

    var A_prev = DenseMatrix.zeros[Double](dimK, dimK)
    var lambda: breeze.linalg.DenseVector[Double] = DenseVector.zeros[Double](dimK)
    var iter: Int = 0

    println("Start ALS iterations...")

    while ((iter == 0) || ((iter < maxIterations) && !AlgebraUtil.isConverged(A_prev, A))) {
      A_prev = A.copy

      // println("Mode A...")
      A = updateALSiteration(fft_sketch_T, B, C, sketcher)
      lambda = AlgebraUtil.colWiseNorm2(A)
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

  private def updateALSiteration(fft_sketch_T: DenseMatrix[Complex],
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

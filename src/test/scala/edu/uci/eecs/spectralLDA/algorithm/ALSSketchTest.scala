package edu.uci.eecs.spectralLDA.algorithm

import breeze.linalg._
import breeze.stats.distributions.{Dirichlet, Multinomial}
import breeze.signal.fourierTr
import breeze.math.Complex
import breeze.numerics.abs
import edu.uci.eecs.spectralLDA.sketch.TensorSketcher
import org.scalatest._
import org.scalatest.Matchers._
import org.apache.spark.SparkContext
import edu.uci.eecs.spectralLDA.utils.TensorOps
import edu.uci.eecs.spectralLDA.testharness.Context

class ALSSketchTest extends FlatSpec with Matchers {

  private val sc: SparkContext = Context.getSparkContext

  "T(C katri-rao dot B) via sketching" should "be close to the exact result" in {
    val k: Int = 20
    val p: DenseVector[Double] = DenseVector.rand(k)
    val t: Tensor[Seq[Int], Double] = Counter()
    for (i <- 0 until k; j <- 0 until k; l <- 0 until k) {
      t(Seq(i, j, l)) = p(i) * p(j) * p(l)
    }

    val sketcher = TensorSketcher[Double, Double](
      n = Seq(k, k, k),
      B = 100,
      b = Math.pow(2, 8).toInt
    )

    val sketch_t = sketcher.sketch(t)
    val fft_sketch_t: DenseMatrix[Complex] = fourierTr(sketch_t(*, ::))

    val C: DenseMatrix[Double] = DenseMatrix.rand(k, k)
    val B: DenseMatrix[Double] = DenseMatrix.rand(k, k)
    val krCB = TensorOps.krprod(C, B)

    val exactResult = TensorOps.unfoldTensor3d(t, Seq(k, k, k)) * krCB
    val sketchResult = TensorSketchOps.TIUV(fft_sketch_t, B, C, sketcher)

    val diff = sketchResult - exactResult
    val norm1 = norm(norm(exactResult(::, *)).toDenseVector)
    val norm2 = norm(norm(diff(::, *)).toDenseVector)
    norm2 should be <= norm1 * 0.2
  }

}
package Accumulator

/**
 * Created by furongh on 11/3/15.
 */
import org.apache.spark.AccumulatorParam

/**
 * Created by furongh on 9/10/15.
 */
object DenseVectorAccumulatorParam extends AccumulatorParam[breeze.linalg.DenseVector[Double]] {
  def zero(initialValue: breeze.linalg.DenseVector[Double]) : breeze.linalg.DenseVector[Double] = {
    breeze.linalg.DenseVector.zeros[Double](initialValue.length)
  }
  def addInPlace(m1: breeze.linalg.DenseVector[Double], m2: breeze.linalg.DenseVector[Double]): breeze.linalg.DenseVector[Double] = {
    m1 += m2
  }
}
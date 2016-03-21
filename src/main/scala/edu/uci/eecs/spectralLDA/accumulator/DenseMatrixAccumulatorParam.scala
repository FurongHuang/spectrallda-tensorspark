package edu.uci.eecs.spectralLDA.accumulator

/**
 * Created by furongh on 11/2/15.
 */

import org.apache.spark.AccumulatorParam

object DenseMatrixAccumulatorParam extends AccumulatorParam[breeze.linalg.DenseMatrix[Double]] {
  def zero(initialValue: breeze.linalg.DenseMatrix[Double]): breeze.linalg.DenseMatrix[Double] = {
    breeze.linalg.DenseMatrix.zeros[Double](initialValue.rows, initialValue.cols)
  }

  def addInPlace(m1: breeze.linalg.DenseMatrix[Double], m2: breeze.linalg.DenseMatrix[Double]): breeze.linalg.DenseMatrix[Double] = {
    m1 += m2
  }
}

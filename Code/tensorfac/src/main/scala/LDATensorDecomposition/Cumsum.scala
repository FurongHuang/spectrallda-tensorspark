package LDATensorDecomposition

/**
 * Created by furongh on 9/15/15.
 */
object Cumsum {

 // def apply(xs : Seq[Int]) : Seq[Int] =
 //   xs.scanLeft(0)(_ + _).tail

  def apply(xs : Array[Double]) : Array[Double] =
    xs.scanLeft(0.0)(_+_).tail
}

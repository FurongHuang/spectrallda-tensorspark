package edu.uci.eecs.spectralLDA.sketch

import breeze.math._
import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import breeze.storage.Zero
import scala.language.postfixOps
import scala.reflect.ClassTag

/** Performs tensor sketching, recovery, and convolution
  *
  * @tparam V value type of the input tensor
  * @tparam W value type of the sign functions \xi and the hashes
  * */
trait TensorSketcherBase[V, W] {
  def sketch(t: Tensor[Seq[Int], V])(implicit ev: V => W): DenseMatrix[W]

  def recover(f: DenseMatrix[W])(implicit ev2: Double => V): Tensor[Seq[Int], V]
}

/** Tensor sketching for any general tensor
  *
  * @param n shape of the tensor [n_1, n_2, ..., n_p]
  * @param b number of hashes
  * @param B number of hash families
  * @param xi the sign functions with norm 1, indexed by (hash_family_id, i, j),
  *           where 1\le i\le p, 1\le j\le n_i
  * @param h the index hash functions, indexed by (hash_family_id, i, j),
  *          where 1\le i\le p, 1\le j\le n_i
  * @tparam V value type of the input tensor
  * @tparam W value type of the sign functions \xi and the hashes
  */
class TensorSketcher[@specialized(Double) V : Numeric : ClassTag : Semiring : Zero,
                     @specialized(Double) W : Numeric : ClassTag : Semiring : Zero]
        (n: Seq[Int],
         b: Int = Math.pow(2, 12).toInt,
         B: Int = 1,
         xi: Tensor[(Int, Int, Int), W],
         h: Tensor[(Int, Int, Int), Int])
  extends TensorSketcherBase[V, W] {
  // order of the tensor
  val p: Int = n.size

  /** Apply the B hash families on a given tensor to sketch */
  override def sketch(a: Tensor[Seq[Int], V])(implicit ev: V => W): DenseMatrix[W] = {
    // We need to perform a check to ensure a's dimensions are equal to n
    // but Breeze has no such support as of 02/2016
    //assert(a.dimensions == n)

    // Generate cartesian product of index ranges along all dimensions
    // [1,n_1]x[1,n_2]x...x[1,n_p], for iteration later on
    val indexRanges = n map { d => Range(0, d) }
    val multiIndexRange = cartesianProduct[Int](indexRanges)

    val evW = implicitly[Numeric[W]]
    import evW._
    val result = DenseMatrix.zeros[W](B, b)

    for (hashFamilyId <- 0 until B; tensorIndex <- multiIndexRange) {
      // For each index from the cartesian space [1,n_1]x[1,n_2]x...x[1,n_p]
      // compute the contribution of the tensor's element to the final hashes
      val hashedIndex: Int = (
        (0 until p) map {
          d => h((hashFamilyId, d, tensorIndex(d)))
        } sum
        ) % b
      val hashedCoeffs = (0 until p) map {
        d => xi((hashFamilyId, d, tensorIndex(d)))
      }

      result(hashFamilyId, hashedIndex) += hashedCoeffs.product * ev(a(tensorIndex))
    }

    result
  }

  override def recover(f: DenseMatrix[W])(implicit ev2: Double => V): Tensor[Seq[Int], V] = {
    val tensor: Tensor[Seq[Int], V] = Counter[Seq[Int], V]()

    // Generate cartesian product of index ranges along all dimensions
    // [1,n_1]x[1,n_2]x...x[1,n_p], for iteration later on
    val indexRanges = n map { d => Range(0, d) }
    val multiIndexRange = cartesianProduct[Int](indexRanges)

    val evW = implicitly[Numeric[W]]
    import evW._

    for (tensorIndex <- multiIndexRange) {
      val seq: Seq[Double] = for {
          hashFamilyId <- 0 until B

          hashedIndex = (
            (0 until p) map {
              d => h((hashFamilyId, d, tensorIndex(d)))
            } sum
            ) % b

          hashedCoeffs = (0 until p) map {
            d => xi((hashFamilyId, d, tensorIndex(d)))
          }
        } yield evW.toDouble(hashedCoeffs.product * f(hashFamilyId, hashedIndex))

      tensor(tensorIndex) = ev2(median(DenseVector(seq: _*)))
    }

    tensor
  }

  /** sketch a matrix specifically */
  def sketch(a: Matrix[V])(implicit ev: V => W): DenseMatrix[W] = {
    require(p == 2)

    val evW = implicitly[Numeric[W]]
    import evW._
    val result = DenseMatrix.zeros[W](B, b)

    for (hashFamilyId <- 0 until B; i <- 0 until a.rows; j <- 0 until a.cols) {
      // For each index from the cartesian space [1,n_1]x[1,n_2]x...x[1,n_p]
      // compute the contribution of the tensor's element to the final hashes
      val hashedIndex = (h((hashFamilyId, 0, i)) + h((hashFamilyId, 1, j))) % b
      val hashedCoeff = xi((hashFamilyId, 0, i)) * xi((hashFamilyId, 1, j))

      result(hashFamilyId, hashedIndex) += hashedCoeff * ev(a(i, j))
    }

    result
  }

  /** Sketch a vector along the given dimension */
  def sketch(v: Vector[V], d: Int)(ev: V => W): DenseMatrix[W] = {
    assert(v.size == n(d))

    val evW = implicitly[Numeric[W]]
    import evW._
    val result = DenseMatrix.zeros[W](B, b)

    for (hashFamilyId <- 0 until B; i <- 0 until n(d)) {
      val hashedIndex: Int = h((hashFamilyId, d, i))
      val hashCoeff: W = xi((hashFamilyId, d, i))
      result(hashFamilyId, hashedIndex) += hashCoeff * ev(v(i))
    }

    result
  }

  private def cartesianProduct[A](xs: Traversable[Traversable[A]]): Seq[Seq[A]] = {
    xs.foldLeft(Seq(Seq.empty[A])){
      (x, y) => for (a <- x.view; b <- y) yield a :+ b
    }
  }
}

object TensorSketcher {
  def apply[@specialized(Double) V : Numeric : ClassTag : Semiring : Zero,
            @specialized(Double) W : Numeric : ClassTag : Semiring : Zero]
          (n: Seq[Int],
           b: Int = Math.pow(2, 12).toInt,
           B: Int = 1,
           kWiseIndependent: Int = 2,
           seed: Option[Int] = None)
          : TensorSketcher[V, W] =  {
    val (xi: Tensor[(Int, Int, Int), W], h: Tensor[(Int, Int, Int), Int]) =
      HashFunctions[W](n, b, B, kWiseIndependent, seed)
    new TensorSketcher[V, W](n, b, B, xi, h)
  }
}

package edu.uci.eecs.spectralLDA.sketch

import breeze.math._
import breeze.linalg._
import breeze.stats.distributions._
import breeze.storage.Zero

/** Generate independent hash functions h and sign functions \xi */
object HashFunctions {
  /** Generate independent hash functions
    *
    * @param n   Length-q Seq (d_1, ..., d_q), q is the order of the tensor, d_i the size along dimension i
    * @param b   Length of each hash
    * @param B   Number of hash families
    * @param kWiseIndependent k-wise independent hash functions, 2 for current implementation
    * @param randBasis    Random seed
    * @tparam W  The space of hashed values, Double or Complex
    * @return    B-by-q-by-d_i tensor for the sign functions \xi, and
    *            B-by-q-by-d_i tensor for the hash functions h
    */
  def apply[@specialized(Double) W : Numeric : Semiring : Zero]
                (n: Seq[Int],
                 b: Int,
                 B: Int,
                 kWiseIndependent: Int = 2
                )
                (implicit randBasis: RandBasis = Rand)
      : (Tensor[(Int, Int, Int), W], Tensor[(Int, Int, Int), Int]) = {
    // The current version only implemented for 2-wise independent hash functions
    for (d <- n) assert(d > 0)
    assert(b > 0)
    assert(B > 0)
    assert(kWiseIndependent == 2)

    val uniform = new Uniform(0, 1)
    val ev = implicitly[Numeric[W]]
    val xiValues: Seq[W] = uniform.sample(B * n.sum) map { r => ev.fromInt((2 * r).toInt * 2 - 1) }
    val hValues: Seq[Int] = uniform.sample(B * n.sum) map { r => (b * r).toInt }

    val xi: Tensor[(Int, Int, Int), W] = Counter[(Int, Int, Int), W]()
    val h: Tensor[(Int, Int, Int), Int] = Counter[(Int, Int, Int), Int]()
    for (hashFamilyId <- 0 until B; d <- 0 until n.size; i <- 0 until n(d)) {
      val l = hashFamilyId * n.sum + n.slice(0, d).sum + i
      xi((hashFamilyId, d, i)) = xiValues(l)
      h((hashFamilyId, d, i)) = hValues(l)
    }

    (xi, h)
  }
}

package edu.uci.eecs.spectralLDA.sketch

import breeze.math._
import breeze.linalg._
import breeze.stats.distributions._
import breeze.storage.Zero
import org.apache.commons.math3.random.MersenneTwister

/** Generate independent hash functions h and sign functions \xi */
object HashFunctions {
  def apply[@specialized(Double) W : Numeric : Semiring : Zero](n: Seq[Int],
                                b: Int,
                                B: Int,
                                kWiseIndependent: Int = 2,
                                seed: Option[Int] = None
                               )
      : (Tensor[(Int, Int, Int), W], Tensor[(Int, Int, Int), Int]) = {
    // The current version only implemented for 2-wise independent hash functions
    assert(kWiseIndependent == 2)

    seed.foreach { sd =>
      implicit val randBasis: RandBasis = new RandBasis(
        new ThreadLocalRandomGenerator(new MersenneTwister(sd))
      )
    }

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

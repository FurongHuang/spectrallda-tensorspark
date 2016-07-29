package edu.uci.eecs.spectralLDA.textprocessing

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

object CombineSmallTextFiles {
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName("Combine Small Text Files")
    val sc = new SparkContext(conf)

    val rdd: RDD[(String, String)] = sc.wholeTextFiles(args.mkString(","))
    rdd.saveAsObjectFile(args(0) + ".obj")
  }
}


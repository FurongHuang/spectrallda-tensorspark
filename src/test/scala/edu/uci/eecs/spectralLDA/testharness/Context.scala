package edu.uci.eecs.spectralLDA.testharness

import org.apache.spark.{SparkConf, SparkContext}

object Context {
  private val conf : SparkConf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("TensorLDASketchTest")
  private val sc: SparkContext = new SparkContext(conf)

  def getSparkContext: SparkContext = {
    sc
  }
}
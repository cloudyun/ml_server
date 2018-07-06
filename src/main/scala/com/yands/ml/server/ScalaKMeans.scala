package com.yands.ml.server

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vectors

class ScalaKMeans {
  
  def train(name: String, mode: String, path: String, k: Int, itera: Int): KMeansModel = {
    val conf = new SparkConf().setMaster(mode).setAppName(name)
    val sc = new SparkContext(conf)
    val data = sc.textFile(path)
    val map = data.map(_.split(" ")).map { x => 
      val values = new Array[Double](3)
      for (a <- 0 until values.length) {
        values(a) = x.apply(a).toDouble
      }
      val dense = Vectors.dense(values)
      dense
    }
    return KMeans.train(map, k, itera);
  }
}
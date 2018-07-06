package com.yands.ml.server

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vectors
import org.springframework.stereotype.Service;

/**  
 * @Title:  ScalaKMeans.scala   
 * @Package com.yands.ml.server   
 * @Description:    (scala版kmeans)   
 * @author: gaoyun     
 * @edit by: 
 * @date:   2018年7月6日 下午6:03:35   
 * @version V1.0 
 */
@Service
class ScalaKMeans {
  
  var sc: SparkContext = null
  
  def train(name: String, mode: String, path: String, k: Int, itera: Int): KMeansModel = {
    val conf = new SparkConf().setMaster(mode).setAppName(name)
    if (sc != null) {
      sc.stop
    }
    sc = new SparkContext(conf)
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
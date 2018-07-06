package com.yands.ml.test

import com.yands.ml.server.ScalaKMeans
import org.apache.spark.mllib.linalg.Vectors

object KMeansTests {
  def main(args: Array[String]): Unit = {
    val name = "test"
    val mode = "local[3]"
    val path = "D:/work/ml_client/target/classes/data/mllib/kmeans_data.txt"
    val model = new ScalaKMeans().train(name, mode, path, 5, 100);
    val values = new Array[Double](3)
    for (a <- 0 until values.length) {
      values(a) = a.toDouble
    }
    val predict = model.predict(Vectors.dense(values))
    print(predict)
  }
}
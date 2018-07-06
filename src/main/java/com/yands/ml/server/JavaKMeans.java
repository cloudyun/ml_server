package com.yands.ml.server;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.springframework.stereotype.Service;

import com.yands.ml.constant.Constant;

/**  
 * @Title:  JavaKMeans.java   
 * @Package com.yands.ml.server   
 * @Description:    (java版kmeans)   
 * @author: gaoyun     
 * @edit by: 
 * @date:   2018年7月6日 下午6:01:03   
 * @version V1.0 
 */ 
@Service
public class JavaKMeans {

	public KMeansModel train(String name, String mode, String path, int k, int itera) {
		SparkConf conf = new SparkConf();
		conf.setAppName(name);
		conf.setMaster(mode);
		Constant.close();
		
		if (Constant.jsc == null) {
			Constant.jsc = new JavaSparkContext(conf);
		}

		JavaRDD<String> data = Constant.jsc.textFile(path);
		JavaRDD<Vector> ratings = data.map(value -> {
			String[] split = value.split(" ");
			double[] values = new double[split.length];
			for (int x = 0; x < split.length; x++) {
				values[x] = Double.parseDouble(split[x]);
			}
			return Vectors.dense(values);
		});
		return KMeans.train(JavaRDD.toRDD(ratings), k, itera);
	}
}

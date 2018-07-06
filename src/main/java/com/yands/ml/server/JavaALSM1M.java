package com.yands.ml.server;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.springframework.stereotype.Service;

import com.yands.ml.constant.Constant;

import scala.Tuple2;

/**  
 * @Title:  JavaALSM1M.java   
 * @Package com.yands.ml.server   
 * @Description:    (java版ALS算法实现)   
 * @author: gaoyun     
 * @edit by: 
 * @date:   2018年7月6日 下午6:00:44   
 * @version V1.0 
 */ 
@Service
public class JavaALSM1M {
	
	public MatrixFactorizationModel train(String name, String mode, int rank, int itera, String path) {
		SparkConf conf = new SparkConf();
		conf.setAppName(name);
		conf.setMaster(mode);
		
		if (Constant.jsc != null) {
			Constant.jsc.close();
			Constant.jsc = null;
		}

		Constant.jsc = new JavaSparkContext(conf);

		JavaRDD<String> data = Constant.jsc.textFile(path);
		JavaRDD<Rating> ratings = data.map(s -> {
			String[] sarray = s.split("\t");
			return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]), Double.parseDouble(sarray[2]));
		});

		return ALS.train(JavaRDD.toRDD(ratings), rank, itera, 0.01);
	}
	
	public Map<Integer, Double> getTopnByUser(MatrixFactorizationModel model, int userId, int topn) {
		Rating[] ts = model.recommendProducts(userId, topn);
//		log.log("用户" + userId + "最喜欢的" + topn + "个商品为:");
		Map<Integer, Double> data = new HashMap<Integer, Double>();
		for (int i = 0; i < ts.length; i++) {
//			log.log("商品:" + ts[i].product() + " 评分:" + ts[i].rating());
			data.put(ts[i].product(), ts[i].rating());
		}
		return data;
	}
	
	public Map<Integer, Double> getTopnByProduct(MatrixFactorizationModel model, int productId, int topn) {
		Rating[] ts = model.recommendUsers(productId, topn);
//		log.log("最喜欢商品" + productId + "的前" + topn + "个用户为:");
		Map<Integer, Double> data = new HashMap<Integer, Double>();
		for (int i = 0; i < ts.length; i++) {
//			log.log("用户:" + ts[i].user() + " 评分:" + ts[i].rating());
			data.put(ts[i].user(), ts[i].rating());
		}
		return data;
	}
	
	public List<Map<String, Object>> verification(MatrixFactorizationModel model, String path) {
		JavaRDD<String> data = Constant.jsc.textFile(path);
		JavaRDD<Tuple2<Integer, Integer>> ratings = data.map(s -> {
			String[] sarray = s.split("\t");
			return new Tuple2<Integer, Integer>(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]));
		});
		List<Map<String, Object>> list = new ArrayList<Map<String, Object>>();
		JavaRDD<Rating> rste = model.predict(JavaPairRDD.fromJavaRDD(ratings));
		List<Rating> collect = rste.sortBy(s -> s.rating(), true, 3).collect();
		
		for (Rating s : collect) {
			Map<String, Object> rat = new HashMap<String, Object>();
			rat.put("user", s.user());
			rat.put("product", s.product());
			rat.put("rating", s.rating());
			list.add(rat);
		}
		return list;
	}
	
	public double forecast(MatrixFactorizationModel model, int userId, int productId) {
//		log.log("用户" + userId + "对商品" + productId + "的预测评分为:" + model.predict(userId, productId));
		return model.predict(userId, productId);
	}
	
	@SuppressWarnings("resource")
	public void main(String[] args) {
		SparkConf conf = new SparkConf();
		conf.setAppName("JavaASLM1M");
		conf.setMaster("local[2]");

		JavaSparkContext jsc = new JavaSparkContext(conf);

		String path = System.class.getResource("/").getPath() + "resources/data/mllib/als/ml-100k/u.data";
		JavaRDD<String> data = jsc.textFile(path);
		JavaRDD<Rating> ratings = data.map(s -> {
			String[] sarray = s.split("\t");
			return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]), Double.parseDouble(sarray[2]));
		});

		/**
		 * 1) ratings : 评分矩阵对应的RDD。需要我们输入。如果是隐式反馈，则是评分矩阵对应的隐式反馈矩阵。
		 * 
		 * 2) rank :
		 * 矩阵分解时对应的低维的维数。即PTm×kQk×nPm×kTQk×n中的维度k。这个值会影响矩阵分解的性能，越大则算法运行的时间和占用的内存可能会越多。通常需要进行调参，一般可以取10-200之间的数。
		 * 
		 * 3) iterations
		 * :在矩阵分解用交替最小二乘法求解时，进行迭代的最大次数。这个值取决于评分矩阵的维度，以及评分矩阵的系数程度。一般来说，不需要太大，比如5-20次即可。默认值是5。
		 * 
		 * 4) lambda: 在
		 * python接口中使用的是lambda_,原因是lambda是Python的保留字。这个值即为FunkSVD分解时对应的正则化系数。主要用于控制模型的拟合程度，增强模型泛化能力。取值越大，则正则化惩罚越强
		 * 。大型推荐系统一般需要调参得到合适的值。
		 * 
		 * 5) alpha :
		 * 这个参数仅仅在使用隐式反馈trainImplicit时有用。指定了隐式反馈信心阈值，这个值越大则越认为用户和他没有评分的物品之间没有关联。一般需要调参得到合适值。
		 */
		int rank = 10;
		int numIterations = 10;
		MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);

		int userId = 1;
		int top = 20;
		Rating[] ts = model.recommendProducts(userId, top);
		System.out.println("用户" + userId + "最喜欢的" + top + "个商品为:");
		for (int i = 0; i < ts.length; i++) {
			System.out.println("商品:" + ts[i].product() + "、评分:" + ts[i].rating());
		}
		
		Rating[] users = model.recommendUsers(20, 12);
		System.out.println("产品" + 20 + "最喜欢的" + 12 + "个用户为:");
		for (int x = 0; x < users.length; x++) {
			System.out.println("用户:" + users[x].user() + "、评分:" + users[x].rating());
		}

		System.out.println("用户18对商品177的预测评分为:" + model.predict(18, 177));

		// 对数据文件u1.base中的数据进行评估
		String pathTest = System.class.getResource("/").getPath() + "resources/data/mllib/als/ml-100k/u1.base";
		JavaRDD<String> data1 = jsc.textFile(pathTest);
		JavaRDD<Tuple2<Integer, Integer>> ratings1 = data1.map(s -> {
			String[] sarray = s.split("\t");
			return new Tuple2<Integer, Integer>(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]));
		});
		JavaRDD<Rating> rste = model.predict(JavaPairRDD.fromJavaRDD(ratings1));
		rste.sortBy(s -> s.rating(), true, 3).foreach(
				s -> 
				System.out.println(s.user() + "\t" + s.product() + "\t" + s.rating())
		);
		
		jsc.close();
	}
}

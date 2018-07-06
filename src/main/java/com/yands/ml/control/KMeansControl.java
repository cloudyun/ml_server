package com.yands.ml.control;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.yands.ml.entity.ResponseData;
import com.yands.ml.server.ScalaKMeans;

@RestController
@RequestMapping("kmeans")
public class KMeansControl {

	private Map<String, KMeansModel> map;
	
	@PostMapping("train")
    public ResponseData train(@RequestBody JSONObject json) {
    	String name = json.getString("name");
		String mode = json.getString("mode");
		String path = json.getString("path");
		int k = json.getIntValue("k");
		int itera = json.getIntValue("itera");
		KMeansModel model = new ScalaKMeans().train(name, mode, path, k, itera);
		if (map == null) {
			map = new HashMap<String, KMeansModel>();
		}
		map.put(name, model);
		return ResponseData.buildNormalMessageResponse("训练成功");
    }
	
	@PostMapping("predict")
    public ResponseData predict(@RequestBody JSONObject json) {
    	String name = json.getString("name");
    	JSONArray arr = json.getJSONArray("vector");
    	double[] vector = new double[arr.size()];
    	for (int x = 0; x < arr.size(); x++) {
    		vector[x] = arr.getDouble(x);
    	}
    	KMeansModel model = map.get(name);
		if (model == null) {
			return ResponseData.buildErrorMessageResponse(303, "请先训练[" + name +"]的数据");
		}
		return ResponseData.buildNormalInfoResponse(model.predict(Vectors.dense(vector)));
    }
}

package com.yands.ml.control;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.alibaba.fastjson.JSONObject;
import com.yands.ml.entity.ResponseData;
import com.yands.ml.server.JavaALSM1M;

/**  
 * @Title:  JavaALSM1MControl.java   
 * @Package com.yands.ml.control   
 * @Description:    (als控制类)   
 * @author: gaoyun     
 * @edit by: 
 * @date:   2018年7月6日 下午6:02:01   
 * @version V1.0 
 */ 
@RestController
@RequestMapping("als")
public class JavaALSM1MControl {
	
	private Map<String, MatrixFactorizationModel> map;

    @Autowired
    private JavaALSM1M javaALSM1M;

    @PostMapping("train")
    public ResponseData train(@RequestBody JSONObject json) {
    	String name = json.getString("name");
		String mode = json.getString("mode");
		int rank = json.getIntValue("rank");
		int itera = json.getIntValue("itera");
		String path = json.getString("path");
		MatrixFactorizationModel model = javaALSM1M.train(name, mode, rank, itera, path);
		if (map == null) {
			map = new HashMap<String, MatrixFactorizationModel>();
		}
		map.put(name, model);
		return ResponseData.buildNormalMessageResponse("训练成功");
    }

    @PostMapping("forecast")
    public ResponseData forecast(@RequestBody JSONObject json) {
    	String name = json.getString("name");
    	int userId = json.getIntValue("userId");
		int productId = json.getIntValue("productId");
		MatrixFactorizationModel model = map.get(name);
		if (model == null) {
			return ResponseData.buildErrorMessageResponse(303, "请先训练[" + name +"]的数据");
		}
		double forecast = javaALSM1M.forecast(model, userId, productId);
		return ResponseData.buildNormalInfoResponse(forecast);
    }

    @PostMapping("getTopnByUser")
    public ResponseData getTopnByUser(@RequestBody JSONObject json) {
    	String name = json.getString("name");
    	int userId = json.getIntValue("userId");
		int topn = json.getIntValue("topn");
		MatrixFactorizationModel model = map.get(name);
		if (model == null) {
			return ResponseData.buildErrorMessageResponse(303, "请先训练[" + name +"]的数据");
		}
		Map<Integer, Double> topnByUser = javaALSM1M.getTopnByUser(model, userId, topn);
		return ResponseData.buildNormalInfoResponse(topnByUser);
    }

    @PostMapping("getTopnByProduct")
    public ResponseData getTopnByProduct(@RequestBody JSONObject json) {
    	String name = json.getString("name");
    	int productId = json.getIntValue("productId");
		int topn = json.getIntValue("topn");
		MatrixFactorizationModel model = map.get(name);
		if (model == null) {
			return ResponseData.buildErrorMessageResponse(303, "请先训练[" + name +"]的数据");
		}
		Map<Integer, Double> topnByProduct = javaALSM1M.getTopnByProduct(model, productId, topn);
		return ResponseData.buildNormalInfoResponse(topnByProduct);
    }

    @PostMapping("verification")
    public ResponseData verification(@RequestBody JSONObject json) {
    	String name = json.getString("name");
    	String path = json.getString("path");
		MatrixFactorizationModel model = map.get(name);
		if (model == null) {
			return ResponseData.buildErrorMessageResponse(303, "请先训练[" + name +"]的数据");
		}
		List<Map<String, Object>> verification = javaALSM1M.verification(model, path);
		return ResponseData.buildNormalInfoResponse(verification);
    }
}

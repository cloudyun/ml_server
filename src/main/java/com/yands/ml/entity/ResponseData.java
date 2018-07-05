package com.yands.ml.entity;

/**
 * gb 返回的统一类型
 */

public class ResponseData {

	private int code = 200;

	private String message ="ok";

	private Object result;

	public int getCode() {
		return code;
	}

	public void setCode(int code) {
		this.code = code;
	}

	public String getMessage() {
		return message;
	}

	public void setMessage(String message) {
		this.message = message;
	}

	public Object getResult() {
		return result;
	}

	public void setResult(Object result) {
		this.result = result;
	}

	public static ResponseData buildNormalResponse(String message, Object info) {
		ResponseData data = new ResponseData();
		data.setMessage(message);
		data.setResult(info);
		return data;
	}

	public static ResponseData buildNormalInfoResponse(Object info) {
		ResponseData data = new ResponseData();
		//data.setMessage("请求成功");
		data.setResult(info);
		return data;
	}

	public static ResponseData buildNormalInfoResponse(Object info, boolean defaultMessage) {
		ResponseData data = new ResponseData();
		data.setMessage("失败");
		if (defaultMessage) {
			data.setMessage("成功");
		}
		data.setResult(info);
		return data;
	}

	public static ResponseData buildNormalMessageResponse(String message) {
		ResponseData data = new ResponseData();
		data.setMessage(message);
		//data.setResult(new HashMap<Object, Object>());
		return data;
	}

	public static ResponseData buildErrorMessageResponse(int code, String message) {
		ResponseData data = new ResponseData();
		data.setCode(code);
		data.setMessage(message);		
		//data.setResult("请求结果异常!!!");
		return data;
	}

	public static ResponseData buildErrorMessageResponse(int code, String message, Object info) {
		ResponseData data = new ResponseData();
		data.setCode(code);
		data.setMessage(message);		
		data.setResult(info);
		return data;
	}

	public static ResponseData buildErrorMessageResponse(int code) {
		ResponseData data = new ResponseData();
		data.setCode(code);
		if(code != 200) {
			data.setMessage("无效代码!!!");			
			data.setResult("无效结果!!!");
		}		
		return data;
	}

	@Override
	public String toString() {
		return "ResponseData{" +
				"code=" + code +
				", message='" + message + '\'' +
				", result=" + result +
				'}';
	}
}

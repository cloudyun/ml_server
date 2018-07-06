# ml_server
本地测试基本环境
  scala:2.10.4 >> 下载官网安装并配置环境变量
  spark:1.6.0 >> 下载官网编译好的包，解压，配置环境变量


打包
  maven clean -e package

测试接口
als

  train:
  url:http://localhost:8899/als/train
  method:POSR
  request_body:{"mode":"local[3]","path":"/D:/work/other/ml.gui/target/classes/data/mllib/als/ml-100k/u.data","itera":10,"name":"test","rank":10}

  response:
  {
    "code": 200,
    "message": "训练成功",
    "result": null
  }

  forcast:
  url:http://localhost:8899/als/forcast
  method:POSR
  request_body:{"userId":10,"name":"test","productId":30}

  response:
  {
    "code": 200,
    "message": "ok",
    "result": 4.567373999851733
  }

kmeans

  train:
  url:http://localhost:8899/kmeans/train
  method:POSR
  request_body:{"mode":"local[3]","path":"D:/work/ml_client/target/classes/data/mllib/kmeans_data.txt","itera":100,"name":"test","k":5}

  response:
  {
    "code": 200,
    "message": "训练成功",
    "result": null
  }

  forcast:
  url:http://localhost:8899/als/predict
  method:POSR
  request_body:{"name":"test", "vector":[2.0, 2.0, 2.0]}

  response:
  {
      "code": 200,
      "message": "ok",
      "result": 2
  }

  

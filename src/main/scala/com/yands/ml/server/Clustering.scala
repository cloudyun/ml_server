package com.yands.ml.server

import breeze.numerics.pow
import breeze.linalg.{DenseVector, sum}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**  
 * @Title:  Clustering.scala   
 * @Package com.yands.ml.server   
 * @Description:    (机器学习)   
 * @author: gaoyun     
 * @edit by: 
 * @date:   2018年7月6日 下午6:03:12   
 * @version V1.0 
 */ 
object Clustering {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf().setMaster("local[4]").setAppName("Clustering")
    val sc = new SparkContext(conf)
    var base_path = System.getProperty("user.dir").toString + "/target/classes/"
    /*加载电影信息*/
    val file_item = sc.textFile(base_path + "data/mllib/als/ml-100k/u.item")
    println(file_item.first())
    /* 1|Toy Story (1995)|01-Jan-1995
    ||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0*/
    /*加载电影类别信息*/
    val file_genre = sc.textFile(base_path + "data/mllib/als/ml-100k/u.genre")
    println(file_genre.first())
    /*加载评论人的信息*/
    val file_user = sc.textFile(base_path + "data/mllib/als/ml-100k/u.user")
    /*加载评论人的评论信息*/
    val file_data = sc.textFile(base_path + "data/mllib/als/ml-100k/u.data")
    /*训练推荐模型*/
    val data_vector = file_data.map(_.split("\t")).map {
      x =>
        Rating(x(0).toInt, x(1).toInt, x(2).toDouble)
    }.cache()
    val aslModel = ALS.train(data_vector, 50, 10, 0.1)
    /*获取用户相似特征*/
    val userFactors = aslModel.userFeatures
    /*用户特征向量化*/
    val userVectors = userFactors.map(x => Vectors.dense(x._2))
    /*获取商品相似特征*/
    val movieFactors = aslModel.productFeatures
    /*商品相似特征向量化*/
    val movieVectors = movieFactors.map(x => Vectors.dense(x._2))
    /*归一化判断*/
    val movieMatrix = new RowMatrix(movieVectors)
    val movieMatrix_Summary = movieMatrix.computeColumnSummaryStatistics()
    println("每列的平均值:" + movieMatrix_Summary.mean) //每列的平均值
    println("每列的方差:" + movieMatrix_Summary.variance) //每列的方差
    val userMatrix = new RowMatrix(userVectors)
    val userMatrix_Summary = userMatrix.computeColumnSummaryStatistics()
    println("每列的平均值:" + userMatrix_Summary.mean) //每列的平均值
    println("每列的方差:" + userMatrix_Summary.variance) //每列的方差
    /*
    [-0.13008311396438857,0.31661349981643944,0.04176194036295573,-0.17211569215642514,-0.4497891125174345,0.09243001925706192,-0.2755917868411943,-0.19225727923869668,-0.018074178428278184,0.043485409688419366,-0.06171785901744992,-0.02591447976817197,0.19002821603433132,-0.05013381354161954,0.15294254500299065,-0.08252016445098687,-0.0057098213683935105,-0.13747174430426554,0.05802203164542926,0.033331425727203726,0.13930257786165984,-0.2941097050176434,-0.19734704657277818,0.1793645468213842,0.1858669768823329,-0.08710850192711901,-0.3094421959292386,-0.2548794820483227,0.05249494735633076,-0.3562774572049559,0.015031007230226604,-0.18366799252050295,-0.08052010999605276,0.23969935994832287,-0.15085203866404293,-0.0266082986122826,0.48041088915071206,0.3819342264520057,0.10262863779907039,0.3689219391143156,0.21978187693621526,-0.04213238444149567,-0.05693009052711689,-0.04565851868781088,0.1643780671247495,-0.05540201890562131,-0.05187046972756212,-0.060050296088081975,-0.021567585541199932,0.4901373003250682]
    [0.03743475409539939,0.031032747373518976,0.029213407748178235,0.03226267596613916,0.040257690707209866,0.034539707321632425,0.027611845740142894,0.028509309121624964,0.025440295510889853,0.02511832158923448,0.03080754975061996,0.03201104062279273,0.02991156881925051,0.0345705916462603,0.035140884173621934,0.031036810909755186,0.03576908913423648,0.03642722793348703,0.029388891832012827,0.02494498821491199,0.03132437145414312,0.0369521012943726,0.02901464695221897,0.027311134522322922,0.02676860109379616,0.032518349865363054,0.028838970048305275,0.0365380120266148,0.04609520070723283,0.05599010197906069,0.03397713010328192,0.029447553057890395,0.028579423940221296,0.030239922213002635,0.03378179513855587,0.02421525892247664,0.046378108876308416,0.03264410698537499,0.03230986943273885,0.051741906473701554,0.035497888204443974,0.02812785680617771,0.025779274246114193,0.031207111744648054,0.03327289703736872,0.028603617535928046,0.040022623716766015,0.029880344948539223,0.02983948000863361,0.04062228040657388]
    [-0.2397596338254717,0.4594882978827614,0.05195971267920803,-0.2277342193865653,-0.6149972658209337,0.12565839801843265,-0.3426050112692244,-0.2757319484961794,0.006145707878818492,0.027304305682852706,-0.05091203888118997,-0.02226309713645111,0.2433257826385377,-0.06589234037518484,0.22518879044694215,-0.0852476117191397,-0.03780259258005953,-0.22765740097208045,0.06361867089908667,0.039030569957985047,0.15821200728387022,-0.39808113600175793,-0.24932252666595706,0.22289011456090352,0.22899166768162324,-0.15776169666472803,-0.3765026197540609,-0.3971351730371645,0.07838055551392697,-0.48751263454885346,0.07657040774049793,-0.24468786257011882,-0.1352783094162975,0.3543083916189498,-0.23719722044210134,-0.06395667672598723,0.7071586536245534,0.4900572946816604,0.11565260993463476,0.5059177976591497,0.28435506896522744,-0.04909286770435483,-0.05596895760600775,-0.05533550545287566,0.19748140330452815,-0.04547192154361222,-0.11179801900008168,-0.04702601122332145,-0.04258431545777241,0.6540003003096279]
    [0.03569838097111229,0.03249774062792991,0.03737983785495416,0.0450460289446052,0.034109606797398676,0.042907070585855724,0.03783857934270285,0.035853469267097295,0.029187533362003956,0.028232712750621707,0.03850051061674194,0.04163192994799209,0.03882657092942245,0.03887707935487142,0.03728989496029858,0.03378360832995031,0.03714453151827954,0.04825414942767919,0.03232008429647026,0.03364460380334661,0.035709645433473214,0.03413621873837159,0.03864382041647934,0.0322039375743242,0.02840916049694515,0.03770926875822772,0.029974874000465297,0.03927937435771869,0.04622604645998578,0.04328527359957934,0.040115015618337774,0.04423951593706847,0.03177280103515792,0.030474791894718824,0.03700057945320223,0.03050651636255259,0.0332588386775746,0.032796935599422934,0.03211705636393917,0.044225120867550045,0.036286898809297856,0.040310338676373035,0.02899783829803054,0.03521037302334469,0.03950475206010323,0.03683043230289064,0.04654706810054013,0.032119433718304606,0.03213251469574535,0.050332188470470975]
    */
    /*对用户K-means因子聚类*/
    val userClusterModel = KMeans.train(userVectors, 5, 100)
    /*使用聚类模型进行预测*/
    val user_predict = userClusterModel.predict(userVectors)
    def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]) = sum(pow(v1 - v2, 2))
    user_predict.map(x => (x, 1)).reduceByKey(_ + _).collect().foreach(println(_))
    /*每个类中的数目
    (4,170)
    (0,230)
    (1,233)
    (2,175)
    (3,135)
    */
    val userInfo = file_user.map(_.split("\\|")).map {
      x => (x(0).toInt, (x(1), x(2), x(3), x(4)))
    }
    /*联合用户信息和特征值*/
    val infoAndFactors = userInfo.join(userFactors)
    val userAssigned = infoAndFactors.map {
      case (userId, ((age, sex, title, zip), factors)) =>
        val pred = userClusterModel.predict(Vectors.dense(factors))
        val center = userClusterModel.clusterCenters(pred)
        val dist = computeDistance(DenseVector(factors), DenseVector(center.toArray))
        (userId, age, sex, title, zip, dist, pred)
    }
    val userCluster = userAssigned.groupBy(_._7).collectAsMap()
    /*输出每个类中的20个用户分类情况*/
    for ((k, v) <- userCluster.toSeq.sortBy(_._1)) {
      println(s"userCluster$k")
      val info = v.toSeq.sortBy(_._6)
      println(info.take(20).map {
        case (userId, age, sex, title, zip, pred, dist) =>
          (userId, age, sex, title, zip)
      }.mkString("\n"))
      println("========================")
    }
    /*
    userCluster0
    (757,26,M,student,55104)
    (276,21,M,student,95064)
    (267,23,M,engineer,83716)
    (643,39,M,scientist,55122)
    (540,28,M,engineer,91201)
    (407,29,M,engineer,03261)
    (135,23,M,student,38401)
    (429,27,M,student,29205)
    (92,32,M,entertainment,80525)
    (624,19,M,student,30067)
    (650,42,M,engineer,83814)
    (70,27,M,engineer,60067)
    (625,27,M,programmer,20723)
    (748,28,M,administrator,94720)
    (292,35,F,programmer,94703)
    (10,53,M,lawyer,90703)
    (26,49,M,engineer,21044)
    (864,27,M,programmer,63021)
    (889,24,M,technician,78704)
    (457,33,F,salesman,30011)
    ========================
    userCluster1
    (916,27,M,engineer,N2L5N)
    (94,26,M,student,71457)
    (645,27,M,programmer,53211)
    (339,35,M,lawyer,37901)
    (666,44,M,administrator,61820)
    (607,49,F,healthcare,02154)
    (85,51,M,educator,20003)
    (543,33,M,scientist,95123)
    (573,68,M,retired,48911)
    (829,48,M,writer,80209)
    (766,42,M,other,10960)
    (184,37,M,librarian,76013)
    (710,19,M,student,92020)
    (794,32,M,educator,57197)
    (60,50,M,healthcare,06472)
    (293,24,M,writer,60804)
    (344,30,F,librarian,94117)
    (360,51,M,other,98027)
    (537,36,M,engineer,22902)
    (18,35,F,other,37212)
    ========================
    userCluster2
    (275,38,M,engineer,92064)
    (554,32,M,scientist,62901)
    (694,60,M,programmer,06365)
    (455,48,M,administrator,83709)
    (178,26,M,other,49512)
    (800,25,M,programmer,55337)
    (738,35,M,technician,95403)
    (488,48,M,technician,21012)
    (647,40,M,educator,45810)
    (764,27,F,educator,62903)
    (87,47,M,administrator,89503)
    (298,44,M,executive,01581)
    (633,35,M,programmer,55414)
    (311,32,M,technician,73071)
    (484,27,M,student,21208)
    (786,36,F,engineer,01754)
    (398,40,M,other,60008)
    (290,40,M,engineer,93550)
    (749,33,M,other,80919)
    (25,39,M,engineer,55107)
    ========================
    userCluster3
    (56,25,M,librarian,46260)
    (552,45,M,other,68147)
    (804,39,M,educator,61820)
    (606,28,M,programmer,63044)
    (33,23,M,student,27510)
    (162,25,M,artist,15610)
    (348,24,F,student,45660)
    (504,40,F,writer,92115)
    (393,19,M,student,83686)
    (545,27,M,technician,08052)
    (826,28,M,artist,77048)
    (396,57,M,engineer,94551)
    (728,58,M,executive,94306)
    (332,20,M,student,40504)
    (320,19,M,student,24060)
    (907,25,F,other,80526)
    (319,38,M,programmer,22030)
    (200,40,M,programmer,93402)
    (923,21,M,student,E2E3R)
    (596,20,M,artist,77073)
    ========================
    userCluster4
    (378,35,M,student,02859)
    (591,57,F,librarian,92093)
    (345,28,F,librarian,94143)
    (106,61,M,retired,55125)
    (594,46,M,educator,M4J2K)
    (908,44,F,librarian,68504)
    (329,48,M,educator,01720)
    (450,35,F,educator,11758)
    (701,51,F,librarian,56321)
    (876,41,M,other,20902)
    (530,29,M,engineer,94040)
    (376,28,F,other,10010)
    (207,39,M,marketing,92037)
    (716,36,F,administrator,44265)
    (84,32,M,executive,55369)
    (271,51,M,engineer,22932)
    (144,53,M,programmer,20910)
    (328,51,M,administrator,06779)
    (297,29,F,educator,98103)
    (262,19,F,student,78264)
    ========================
    */ 
    /*对电影K-means因子聚类*/
    val movieClusterModel = KMeans.train(movieVectors, 5, 100)
    /*KMeans: KMeans converged in 39 iterations.*/
    val movie_predict = movieClusterModel.predict(movieVectors)
    movie_predict.map(x => (x, 1)).reduceByKey(_ + _).collect.foreach(println(_))
    /*result
    (4,384)
    (0,340)
    (1,154)
    (2,454)
    (3,350)
     */
    /*查看及分析商品相似度聚类数据*/
    /*提取电影的题材标签*/
    val genresMap = file_genre.filter(!_.isEmpty).map(_.split("\\|"))
      .map(x => (x(1), x(0))).collectAsMap()
    /*为电影数据和题材映射关系创建新的RDD，其中包含电影ID、标题和题材*/
    val titlesAndGenres = file_item.map(_.split("\\|")).map {
      array =>
        val geners = array.slice(5, array.size).zipWithIndex.filter(_._1 == "1").map(
          x => genresMap(x._2.toString))
        (array(0).toInt, (array(1), geners))
    }
    val titlesWithFactors = titlesAndGenres.join(movieFactors)
    val movieAssigned = titlesWithFactors.map {
      case (id, ((movie, genres), factors)) =>
        val pred = movieClusterModel.predict(Vectors.dense(factors))
        val center = movieClusterModel.clusterCenters(pred)
        val dist = computeDistance(DenseVector(factors), DenseVector(center.toArray))
        (id, movie, genres.mkString(" "), pred, dist)
    }
    val clusterAssigned = movieAssigned.groupBy(_._4).collectAsMap()
    for ((k, v) <- clusterAssigned.toSeq.sortBy(_._1)) {
      println(s"Cluster$k")
      val dist = v.toSeq.sortBy(_._5)
      println(dist.take(20).map {
        case (id, movie, genres, pred, dist) =>
          (id, movie, genres)
      }.mkString("\n"))
      println("============")
    }
    /*
    Cluster0
    (1123,Last Time I Saw Paris, The (1954),Drama)
    (1526,Witness (1985),Drama Romance Thriller)
    (711,Substance of Fire, The (1996),Drama)
    (1674,Mamma Roma (1962),Drama)
    (1541,Beans of Egypt, Maine, The (1994),Drama)
    (1454,Angel and the Badman (1947),Western)
    (1537,Cosi (1996),Comedy)
    (1506,Nelly & Monsieur Arnaud (1995),Drama)
    (483,Casablanca (1942),Drama Romance War)
    (479,Vertigo (1958),Mystery Thriller)
    (1627,Wife, The (1995),Comedy Drama)
    (513,Third Man, The (1949),Mystery Thriller)
    (608,Spellbound (1945),Mystery Romance Thriller)
    (1122,They Made Me a Criminal (1939),Crime Drama)
    (1124,Farewell to Arms, A (1932),Romance War)
    (1525,Object of My Affection, The (1998),Comedy Romance)
    (1573,Spirits of the Dead (Tre passi nel delirio) (1968),Horror)
    (58,Quiz Show (1994),Drama)
    (505,Dial M for Murder (1954),Mystery Thriller)
    (1460,Sleepover (1995),Comedy Drama)
    ============
    Cluster1
    (1455,Outlaw, The (1943),Western)
    (54,Outbreak (1995),Action Drama Thriller)
    (281,River Wild, The (1994),Action Thriller)
    (1668,Wedding Bell Blues (1996),Comedy)
    (1670,Tainted (1998),Comedy Thriller)
    (1667,Next Step, The (1995),Drama)
    (1657,Target (1995),Action Drama)
    (1477,Nightwatch (1997),Horror Thriller)
    (870,Touch (1997),Romance)
    (1430,Ill Gotten Gains (1997),Drama)
    (918,City of Angels (1998),Romance)
    (1249,For Love or Money (1993),Comedy)
    (801,Air Up There, The (1994),Comedy)
    (1519,New Jersey Drive (1995),Crime Drama)
    (619,Extreme Measures (1996),Drama Thriller)
    (1613,Tokyo Fist (1995),Action Drama)
    (1542,Scarlet Letter, The (1926),Drama)
    (576,Cliffhanger (1993),Action Adventure Crime)
    (808,Program, The (1993),Action Drama)
    (471,Courage Under Fire (1996),Drama War)
    ============
    Cluster2
    (1539,Being Human (1993),Drama)
    (1371,Machine, The (1994),Comedy Horror)
    (1365,Johnny 100 Pesos (1993),Action Drama)
    (1350,Crows and Sparrows (1949),Drama)
    (1676,War at Home, The (1996),Drama)
    (1513,Sprung (1997),Comedy)
    (1414,Coldblooded (1995),Action)
    (1354,Venice/Venice (1992),Drama)
    (897,Time Tracers (1995),Action Adventure Sci-Fi)
    (1374,Falling in Love Again (1980),Comedy)
    (1334,Somebody to Love (1994),Drama)
    (1359,Boys in Venice (1996),Drama)
    (437,Amityville 1992: It's About Time (1992),Horror)
    (439,Amityville: A New Generation (1993),Horror)
    (1318,Catwalk (1995),Documentary)
    (1360,Sexual Life of the Belgians, The (1994),Comedy)
    (1320,Homage (1995),Drama)
    (1340,Crude Oasis, The (1995),Romance)
    (1364,Bird of Prey (1996),Action)
    (1352,Shadow of Angels (Schatten der Engel) (1976),Drama)
    ============
    Cluster3
    (1223,King of the Hill (1993),Drama)
    (1538,All Over Me (1997),Drama)
    (1370,I Can't Sleep (J'ai pas sommeil) (1994),Drama Thriller)
    (1682,Scream of Stone (Schrei aus Stein) (1991),Drama)
    (1632,Land and Freedom (Tierra y libertad) (1995),War)
    (1640,Eighth Day, The (1996),Drama)
    (1641,Dadetown (1995),Documentary)
    (1649,Big One, The (1997),Comedy Documentary)
    (1633,� k�ldum klaka (Cold Fever) (1994),Comedy Drama)
    (1637,Girls Town (1996),Drama)
    (1630,Silence of the Palace, The (Saimt el Qusur) (1994),Drama)
    (1638,Normal Life (1996),Crime Drama)
    (1635,Two Friends (1986) ,Drama)
    (1647,Hana-bi (1997),Comedy Crime Drama)
    (1356,Ed's Next Move (1996),Comedy)
    (1515,Wings of Courage (1995),Adventure Romance)
    (1423,Walking Dead, The (1995),Drama War)
    (1619,All Things Fair (1996),Drama)
    (1482,Gate of Heavenly Peace, The (1995),Documentary)
    (1578,Collectionneuse, La (1967),Drama)
    ============
    Cluster4
    (1603,Angela (1995),Drama)
    (1521,Mr. Wonderful (1993),Comedy Romance)
    (1096,Commandments (1997),Romance)
    (1441,Moonlight and Valentino (1995),Drama Romance)
    (1516,Wedding Gift, The (1994),Drama)
    (1543,Johns (1996),Drama)
    (1436,Mr. Jones (1993),Drama Romance)
    (1611,Intimate Relations (1996),Comedy)
    (1673,Mirage (1995),Action Thriller)
    (1189,Prefontaine (1997),Drama)
    (625,Sword in the Stone, The (1963),Animation Children's)
    (1285,Princess Caraboo (1994),Drama)
    (1145,Blue Chips (1994),Drama)
    (28,Apollo 13 (1995),Action Drama Thriller)
    (196,Dead Poets Society (1989),Drama)
    (167,Private Benjamin (1980),Comedy)
    (1248,Blink (1994),Thriller)
    (164,Abyss, The (1989),Action Adventure Sci-Fi Thriller)
    (732,Dave (1993),Comedy Romance)
    (1406,When Night Is Falling (1995),Drama Romance)
    */ 
    /*内部评价指标，计算性能*/
    val movieCost = movieClusterModel.computeCost(movieVectors)
    println(movieCost)
    val userCost = movieClusterModel.computeCost(userVectors)
    println(userCost)
    /*
    2296.979970714412
    1719.2302334162873
    */
  }
}
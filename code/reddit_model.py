from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

# IMPORT OTHER MODULES HERE
import re
import string
import itertools
from itertools import chain
from pyspark.sql.types import *

# IMPORT ML RELATED
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric


def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing:
    1. The unigrams
    2. The bigrams
    3. The trigrams
    """
    grams = []
    cleantext = __import__("cleantext")
    res = cleantext.sanitize(text)
    return res[1:]

def union_grams(gramlist):
    """Do union of the text return String
    """
    res = ""
    for item in gramlist:
        res = res + item + " "
    return res[:-1].split()

def remove_t3_f(link_id):
    # remove "t3_f"
    if link_id.startswith("t3_"):
        return link_id[3:]
    return link_id

def task9_remove(text):
    # mark "&gt" to assist filtering
    if text.strip().startswith("&gt"):
        return "&gt"
    text = re.sub(r"/s", "", text)
    return text


def main(context):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED
    spark = sqlContext

    schema = StructType([
    StructField("id", StringType()),
    StructField("dem", IntegerType()),
    StructField("gop", IntegerType()),
    StructField("djt", IntegerType())
	])

    comments = sqlContext.read.json("comments-minimal.json.bz2")
    submissions = sqlContext.read.json("submissions.json.bz2")
    labeled_data = sqlContext.read.load("labeled_data.csv", format="csv", schema=schema, header="true")
	
    comments.select("*").write.save("comments.parquet", format="parquet")
    submissions.select("*").write.save("submissions.parquet", format="parquet")
    labeled_data.select("*").write.save("label.parquet", format="parquet")
    
    comments = spark.read.load("comments.parquet")
    submissions = spark.read.load("submissions.parquet")
    labeled_data = sqlContext.read.load("label.parquet")


    comments.createOrReplaceTempView("commentsView")
    submissions.createOrReplaceTempView("submissionsView")
    labeled_data.createOrReplaceTempView("labeled_dataView")

    sqlDF = spark.sql("SELECT * FROM labeled_dataView l JOIN commentsView c ON l.id = c.id")
    sqlDF.show()

    test = spark.sql("SELECT cleanTextWithPython(body) as grams FROM commentsView")
    test.show()

    res = spark.sql("SELECT unionTextWithPython(cleanTextWithPython(c.body)) as grams, l.dem, l.gop, l.djt FROM commentsView c, labeled_dataView l where c.id = l.id")
    res.show()

    cv = CountVectorizer(inputCol="grams", outputCol="features", binary=True, minDF=5.0)
    model = cv.fit(res)
    result = model.transform(res)
    result.show()

    result.createOrReplaceTempView("resultView")
    ans = spark.sql("SELECT features, IF(djt = 1, 1, 0) as positive,  IF(djt = -1, 1, 0) as negative from resultView")

    ans.createOrReplaceTempView("pos1")
    ans.createOrReplaceTempView("neg1")

    pos = spark.sql("SELECT features, positive as label from pos1")
    neg = spark.sql("SELECT features, negative as label from neg1")


    poslr = ( LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).setThresholds([0.8, 0.2]).setThreshold(0.2) )
    neglr = ( LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).setThresholds([0.75, 0.25]).setThreshold(0.25) )
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator()
    negEvaluator = BinaryClassificationEvaluator()
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=5)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=5)
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    # Train the models

    print("Distribution of Pos and Neg in trainingData is: ", posTrain.groupBy("label").count().take(3))
    print("Distribution of Pos and Neg in testData is: ", posTest.groupBy("label").count().take(3))


    # print("Training positive classifier...")
    # posModel = posCrossval.fit(posTrain)
    # print("Training negative classifier...")
    # negModel = negCrossval.fit(negTrain)

    # # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    # posModel.save("pos.model")
    # negModel.save("neg.model")

    posModel = CrossValidatorModel.load('pos.model')
    negModel = CrossValidatorModel.load('neg.model')

    posTest_res = posModel.transform(posTest)
    negTest_res = negModel.transform(negTest)

    posTest_res.createOrReplaceTempView("posTest_res1")
    spark.sql("SELECT * from posTest_res1 where label <> 0.0").show(50)


    results = posTest_res.select(['probability', 'label']) 
    ## prepare score-label set
    results_collect = results.collect()
    results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
    scoreAndLabels = sc.parallelize(results_list)
     
    metrics = metric(scoreAndLabels)
    print("The ROC score is: ", metrics.areaUnderROC)

    task8 = spark.sql("SELECT s.id, c.created_utc, s.title, c.author_flair_text, c.body FROM commentsView c join submissionsView s on removeheadWithPython(c.link_id) = s.id")
    task8.createOrReplaceTempView("task8_view")
    test_res = spark.sql("SELECT *, unionTextWithPython(cleanTextWithPython(removeSthWithPython(body))) as grams FROM task8_view where removeSthWithPython(body) <> '&gt' ")

    # test_result = model.transform(test_res)
    # test_result.show()
    # df1 = test_result.sample(False, 0.0001, None)
    # posResult = posModel.transform(df1)
    # posResult.createOrReplaceTempView("posResult1")
    # spark.sql("SELECT count(*) FROM posResult1 group By prediction having prediction = 1").show()

if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    # Define UDF for SQL
    sqlContext.udf.register("cleanTextWithPython", sanitize)
    sqlContext.udf.register("unionTextWithPython", union_grams, ArrayType(StringType()))
    sqlContext.udf.register("removeheadWithPython", remove_t3_f)
    sqlContext.udf.register("removeSthWithPython", task9_remove)
    main(sqlContext)


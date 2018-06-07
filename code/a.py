from pyspark.sql import SQLContext

import itertools
from itertools import chain

from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric

from pyspark.sql.functions import udf, col

import re
import string



def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing:
    1. The unigrams
    2. The bigrams
    3. The trigrams
    """
    bounding_punctuation = [".", "!", ":", ",", ";", "?"]

    context = text
    #context = "I'm afraid I can't explain myself, sir. Because I am not myself, you see?"
    #context = peek[0]
    context = re.sub(r"\t\n", " ", context)
    context = re.sub(r"http\S+", "", context)
    context = re.sub(r"\s{2,}", " ", context)
    context = re.findall(r"[\w'/\-%$]+|[.,!?;:]", context)

    context = ' '.join(context).lower()

    context = context.replace("0 , 0", "0,0")
    context = context.replace("i . e", "i.e")
    context = context.replace("e . g", "e.g")

    ############parsed_text############
    parsed_text = context
    words = context.split()

    ############unigrams############
    unigram = []
    i = 0
    while i < len(words):
        if words[i] not in bounding_punctuation:
            unigram.append(words[i])
        i += 1
    unigrams = ' '.join(unigram)

    ############bigrams############
    bigram = []
    i = 1
    while i < len(words):
        if words[i - 1] not in bounding_punctuation \
        and words[i] not in bounding_punctuation:
            bigram.append(words[i - 1] + "_" + words[i])
        i += 1
    bigrams = ' '.join(bigram)
    #print(bigrams)

    ############trigram############
    trigram = []
    i = 2
    while i < len(words):
        if words[i - 2] not in bounding_punctuation\
        and words[i - 1] not in bounding_punctuation\
        and words[i] not in bounding_punctuation:
            trigram.append(words[i - 2] + "_" + words[i - 1] + "_" + words[i])
        i += 1
    trigrams = ' '.join(trigram)
    #print(trigrams)

    return [unigrams, bigrams, trigrams]

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


sqlContext = SQLContext(sc)

schema = StructType([
    StructField("id", StringType()),
    StructField("dem", IntegerType()),
    StructField("gop", IntegerType()),
    StructField("djt", IntegerType())
])

labeled_data = sqlContext.read.load("labeled_data.csv", format="csv", schema=schema, header="true")

comments = spark.read.load("comments.parquet")
submissions = spark.read.load("submissions.parquet")

comments.createOrReplaceTempView("comments1")
labeled_data.createOrReplaceTempView("labeled_data1")

sqlDF = spark.sql("SELECT * FROM labeled_data1 JOIN comments1 on labeled_data1.id = comments1.id")
sqlDF.show()

sqlContext.udf.register("cleanTextWithPython", sanitize)
sqlContext.udf.register("unionTextWithPython", union_grams, ArrayType(StringType()))

res = spark.sql("SELECT unionTextWithPython(cleanTextWithPython(c.body)) as grams, l.dem, l.gop, l.djt FROM comments1 c, labeled_data1 l where c.id = l.id")
res1 = spark.sql("SELECT cleanTextWithPython(c.body) as grams, l.dem, l.gop, l.djt FROM comments1 c, labeled_data1 l where c.id = l.id")

res1.createOrReplaceTempView("res1")
ans1 = spark.sql("SELECT grams, IF(djt = 1, 1, 0) as positive,  IF(djt = -1, 1, 0) as negative from res1")

cv = CountVectorizer(inputCol="grams", outputCol="features", binary=True, minDF=5.0)
model = cv.fit(res)
result = model.transform(res)

result.createOrReplaceTempView("result1")
ans = spark.sql("SELECT features, IF(djt = 1, 1, 0) as positive,  IF(djt = -1, 1, 0) as negative from result1")

ans.createOrReplaceTempView("pos1")
ans.createOrReplaceTempView("neg1")

pos = spark.sql("SELECT features, positive as label from pos1")
neg = spark.sql("SELECT features, negative as label from neg1")


posModel = CrossValidatorModel.load('pos.model')
negModel = CrossValidatorModel.load('neg.model')

comments.createOrReplaceTempView("comments2")
submissions.createOrReplaceTempView("submissions2")

# testfuc = spark.sql("SELECT count(*) from comments2 c join submissions2 s on removeheadWithPython(c.link_id) = s.id") 

sqlContext.udf.register("removeheadWithPython", remove_t3_f)
task8 = spark.sql("SELECT s.id, from_unixtime(c.created_utc, 'YYYYMMdd') as ts, s.title, c.author_flair_text, c.score as c_score, s.score as s_score, c.body FROM comments2 c join submissions2 s on removeheadWithPython(c.link_id) = s.id")
sqlContext.udf.register("removeSthWithPython", task9_remove)
task8.createOrReplaceTempView("task8_view")
test_res = spark.sql("SELECT *, unionTextWithPython(cleanTextWithPython(removeSthWithPython(body))) as grams FROM task8_view where removeSthWithPython(body) <> '&gt' ")

test_result = model.transform(test_res)
df1 = test_result.sample(False, 0.2, None)

posResult = posModel.transform(df1)
negResult = negModel.transform(df1)

posResult.createOrReplaceTempView("df_p")
negResult.createOrReplaceTempView("df_n")

final = spark.sql("SELECT p.id, p.ts, p.title, p.author_flair_text, p.c_score, p.s_score, p.body, p.prediction as pos, n.prediction as neg FROM df_p p, df_n n WHERE p.id = n.id")

final_pos = spark.sql("SELECT id, ts, title, author_flair_text, c_score, s_score, body, prediction FROM df_p")
final_neg = spark.sql("SELECT id, ts, title, author_flair_text, c_score, s_score, body, prediction FROM df_n")


final_pos.createOrReplaceTempView("final_posView")
final_neg.createOrReplaceTempView("final_negView")


top_title_p = spark.sql("SELECT title, sum(prediction)/count(prediction) as percentage FROM final_posView GROUP BY title ORDER BY percentage DESC LIMIT 10")
top_title_n = spark.sql("SELECT title, sum(prediction)/count(prediction) as percentage FROM final_negView GROUP BY title ORDER BY percentage DESC LIMIT 10")

top_title_p.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("top_title_pos.csv")
top_title_n.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("top_title_neg.csv")


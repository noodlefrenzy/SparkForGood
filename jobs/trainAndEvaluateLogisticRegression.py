"""trainingLogisticRegression.py"""
import json
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import SQLContext

sc = SparkContext('local', 'trainAndEvaluateLogisticRegression')
sqlContext = SQLContext(sc)

lines = sc.textFile('../data/tweets/*.json')
sentences = lines.map(lambda x: json.loads(x)['Sentence'])

labeled = sentences.map(lambda x: (1.0 if x.find('hillary') >= 0 and x.find('benghazi') >= 0 else 0.0, x))
sentenceData = sqlContext.createDataFrame(labeled, ['label', 'sentence']).cache()

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2000)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)

# Split the data into training, test using the handy randomSplit.
# It takes a seed if you want to make it repeatable.
trainData, testData = sentenceData.randomSplit([0.8, 0.2])

lr = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.8)
# Let's create a pipeline that tokenizes, hashes,
#  rescales using the TF/IDF we learned, and runs a logistic regressor
lrPipeline = Pipeline(stages=[tokenizer, hashingTF, idfModel, lr])
# Fit it to the training data
lrModel = lrPipeline.fit(trainData)
# Now is the time in Scala where we go lrModel.save('wasb:///models/myLR.model')
# No luck (yet) in Python

lrPredictions = lrModel.transform(testData)

metrics = BinaryClassificationMetrics(lrPredictions.map(lambda x: (x['prediction'], x['label'])))
print("Area under PR = %s" % metrics.areaUnderPR)
print("Area under ROC = %s" % metrics.areaUnderROC)

def cat(x, threshold=0.5):
    if x['label'] > threshold:
        return 'TP' if x['prediction'] > threshold else 'FN'
    else:
        return 'TN' if x['prediction'] <= threshold else 'FP'
posneg = lrPredictions.map(lambda x: (cat(x), x))
posneg.persist()
rates = posneg.countByKey()
precision = rates['TP'] / float(rates['TP'] + rates['FP'])
recall = rates['TP'] / float(rates['TP'] + rates['FN'])
print(rates)
print('Precision (%f) and Recall (%f)' % (precision, recall))

falseVals = posneg.flatMap(lambda x: [ (x[0], x[1]['sentence']) ] if x[0] == 'FP' or x[0] == 'FN' else []).distinct()
for fr in falseVals.take(100):
	print(fr[0] + ': ' + fr[1])


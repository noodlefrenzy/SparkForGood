"""featurizingWithTFIDF.py"""
import json
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SQLContext

sc = SparkContext('local', 'featurizingWithTFIDF')
sqlContext = SQLContext(sc)

lines = sc.textFile('../data/tweets/*.json')
sentences = lines.map(lambda x: json.loads(x)['Sentence'])

labeled = sentences.map(lambda x: (1.0 if x.find('hillary') >= 0 and x.find('benghazi') >= 0 else 0.0, x))
sentenceData = sqlContext.createDataFrame(labeled, ['label', 'sentence'])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2000)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
print(rescaledData.head())
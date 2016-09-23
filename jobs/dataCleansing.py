"""dataCleansing.py"""
import json
from pyspark import SparkContext

sc = SparkContext('local', 'dataCleansing')

lines = sc.textFile('../data/tweets/*.json')
kws = lines.map(json.loads).flatMap(lambda x: [] if 'Keywords' not in x else [(kw, 1) for kw in x['Keywords']])
histo = kws.reduceByKey(lambda x, y: x + y)
for kv in histo.takeOrdered(10, key=lambda x: -x[1]):
	print('Keyword "%s": %i' % kv)

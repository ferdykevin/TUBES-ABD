from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Check Gold") \
    .getOrCreate()

df = spark.read.parquet("lakehouse/gold/analytics_ready")

df.show(10)

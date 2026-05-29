from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("DBD Prediction") \
    .getOrCreate()

# Read Gold Layer

df = spark.read.parquet(
    "lakehouse/gold/final_analytics"
)

# Feature columns

feature_columns = [
    "curah_hujan",
    "suhu",
    "kelembaban",
    "kepadatan_penduduk"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

final_data = assembler.transform(df)

# Split

train_data, test_data = final_data.randomSplit([0.8, 0.2])

# Random Forest

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="risiko_dbd",
    numTrees=20
)

model = rf.fit(train_data)

predictions = model.transform(test_data)

# Evaluation

evaluator = MulticlassClassificationEvaluator(
    labelCol="risiko_dbd",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy}")

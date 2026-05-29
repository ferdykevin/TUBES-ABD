from pyspark.sql import SparkSession
from pyspark.sql.functions import when

spark = SparkSession.builder \
    .appName("Gold Layer") \
    .getOrCreate()

# =========================
# READ SILVER LAYER
# =========================

df = spark.read.parquet(
    "lakehouse/silver/cleaned_data"
)

# =========================
# FEATURE ENGINEERING
# =========================

df = df.withColumn(
    "risk_label",
    when(df.total_cases > 50, 1).otherwise(0)
)

# =========================
# SAVE GOLD LAYER
# =========================

df.write.mode("overwrite").parquet(
    "lakehouse/gold/analytics_ready"
)

print("Gold Layer Created Successfully")

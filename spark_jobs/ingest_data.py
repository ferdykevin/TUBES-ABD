from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Bronze Layer") \
    .getOrCreate()

# =========================
# LOAD FEATURES DATA
# =========================

features_df = spark.read.csv(
    "data/raw/features.csv",
    header=True,
    inferSchema=True
)

# =========================
# LOAD LABELS DATA
# =========================

labels_df = spark.read.csv(
    "data/raw/labels.csv",
    header=True,
    inferSchema=True
)

# =========================
# LOAD POPULATION DATA
# =========================

population_df = spark.read.csv(
    "data/raw/population.csv",
    header=True,
    inferSchema=True
)

# =========================
# SAVE TO BRONZE LAYER
# =========================

features_df.write.mode("overwrite").parquet(
    "lakehouse/bronze/features"
)

labels_df.write.mode("overwrite").parquet(
    "lakehouse/bronze/labels"
)

population_df.write.mode("overwrite").parquet(
    "lakehouse/bronze/population"
)

print("Bronze Layer Created Successfully")

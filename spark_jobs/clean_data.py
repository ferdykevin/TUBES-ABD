from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Silver Layer") \
    .getOrCreate()

# =========================
# READ BRONZE LAYER
# =========================

features = spark.read.parquet(
    "lakehouse/bronze/features"
)

labels = spark.read.parquet(
    "lakehouse/bronze/labels"
)

population = spark.read.parquet(
    "lakehouse/bronze/population"
)

# =========================
# JOIN FEATURES + LABELS
# =========================

joined_df = features.join(
    labels,
    ["city", "year", "weekofyear"],
    "inner"
)

# =========================
# JOIN POPULATION
# =========================

final_df = joined_df.join(
    population,
    ["city", "year"],
    "left"
)

# =========================
# CLEAN NULL VALUES
# =========================

final_df = final_df.dropna()

# =========================
# SAVE SILVER LAYER
# =========================

final_df.write.mode("overwrite").parquet(
    "lakehouse/silver/cleaned_data"
)

print("Silver Layer Created Successfully")

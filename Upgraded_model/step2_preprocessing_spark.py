# ============================================================
# PHYSICAL EDUCATION ASSESSMENT
# STEP 2 : DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------

import os
import findspark
findspark.init()           # Initialize Spark environment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum, rand, least, greatest, lit
)
from pyspark.ml.feature import VectorAssembler, MinMaxScaler


# ------------------------------------------------------------
# 1. DEFINE CONSTANT PATHS AND DIRECTORIES
# ------------------------------------------------------------

DATA_PATH = "Upgraded_model/data/pe_assessment_dataset.csv"
OUTPUT_VIS_DIR = "Upgraded_model/visualizations"
OUTPUT_DATA_DIR = "data"

# Create directories if they do not exist
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. START SPARK SESSION
# ------------------------------------------------------------

spark = SparkSession.builder \
    .appName("PE_Assessment_Data_Preprocessing") \
    .getOrCreate()

print("Spark Version Used :", spark.version)


# ------------------------------------------------------------
# 3. LOAD DATASET INTO SPARK DATAFRAME
# ------------------------------------------------------------

df = spark.read.csv(
    DATA_PATH,
    header=True,
    inferSchema=True
)

# Display dataset information
print("Total number of records :", df.count())
df.printSchema()


# ------------------------------------------------------------
# 4. FEATURE ENGINEERING
#    Creation of Psycho-Social Attributes
# ------------------------------------------------------------
# These features simulate real-life student behaviour
# Values are bounded using least() and greatest()
# ------------------------------------------------------------

df = df.withColumn(
    "motivation",
    least(
        greatest(col("participation") / 10 + rand() * 2, lit(4)),
        lit(9)
    )
)

df = df.withColumn(
    "stress_level",
    least(
        greatest(8 - col("physical_progress") / 15 + rand() * 1.5, lit(2)),
        lit(8)
    )
)

df = df.withColumn(
    "self_confidence",
    least(
        greatest(col("physical_progress") / 10 + rand() * 2, lit(4)),
        lit(9)
    )
)

df = df.withColumn(
    "focus",
    least(
        greatest(8 - col("stress_level") / 2 + rand(), lit(3)),
        lit(9)
    )
)

df = df.withColumn(
    "teamwork",
    least(
        greatest(col("participation") / 10 + rand() * 2, lit(4)),
        lit(9)
    )
)

df = df.withColumn(
    "peer_support",
    least(
        greatest(lit(6) + rand() * 3, lit(5)),
        lit(9)
    )
)

df = df.withColumn(
    "communication",
    least(
        greatest(col("teamwork") + rand(), lit(4)),
        lit(9)
    )
)

df = df.withColumn(
    "sleep_quality",
    least(
        greatest(col("endurance") / 10 + rand() * 2, lit(4)),
        lit(9)
    )
)

df = df.withColumn(
    "nutrition",
    least(
        greatest(col("strength") / 10 + rand() * 2, lit(4)),
        lit(9)
    )
)

df = df.withColumn(
    "screen_time",
    least(
        greatest(8 - col("focus") / 2 + rand(), lit(2)),
        lit(8)
    )
)

# Display summary statistics of engineered features
df.select(
    "motivation", "stress_level", "self_confidence",
    "focus", "teamwork", "sleep_quality", "screen_time"
).summary().show()


# ------------------------------------------------------------
# 5. REMOVE DUPLICATE COLUMNS (SAFETY STEP)
# ------------------------------------------------------------

# Ensures no duplicate column names exist
df = df.select(*list(dict.fromkeys(df.columns)))


# ------------------------------------------------------------
# 6. REMOVE INVALID ZERO RECORDS
# ------------------------------------------------------------
# Rows where all performance-related attributes are zero
# are considered invalid and removed
# ------------------------------------------------------------

df = df.filter(
    (
        col("attendance") +
        col("endurance") +
        col("strength") +
        col("flexibility") +
        col("participation") +
        col("skill_speed") +
        col("physical_progress") +
        col("score")
    ) > 0
)

# Cache dataframe for performance
df = df.cache()

print("Valid records after cleaning :", df.count())


# ------------------------------------------------------------
# 7. CHECK AND REMOVE MISSING VALUES
# ------------------------------------------------------------

# Display null count per column
df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show()

# Drop rows with missing values
df = df.dropna()


# ------------------------------------------------------------
# 8. CONVERT SAMPLE DATA TO PANDAS
#    (USED ONLY FOR VISUALIZATION)
# ------------------------------------------------------------

pdf = df.sample(fraction=0.1, seed=42) \
        .limit(10000) \
        .toPandas()


# ------------------------------------------------------------
# 9. DATA VISUALIZATION
# ------------------------------------------------------------

features = [
    "attendance", "endurance", "strength", "flexibility",
    "participation", "skill_speed", "physical_progress",
    "motivation", "stress_level", "self_confidence",
    "focus", "teamwork", "peer_support", "communication",
    "sleep_quality", "nutrition", "screen_time"
]

# ---------- Feature Distributions ----------
columns = 4
rows = int(np.ceil(len(features) / columns))

plt.figure(figsize=(columns * 4, rows * 3))
pdf[features].hist(bins=20, layout=(rows, columns))
plt.suptitle("Distribution of Physical and Psycho-Social Features")
plt.tight_layout()
plt.savefig(f"{OUTPUT_VIS_DIR}/feature_distributions.png")
plt.close()

# ---------- Score Distribution ----------
plt.figure(figsize=(7, 5))
plt.hist(pdf["score"], bins=25)
plt.xlabel("PE Score")
plt.ylabel("Number of Students")
plt.title("Distribution of PE Assessment Scores")
plt.tight_layout()
plt.savefig(f"{OUTPUT_VIS_DIR}/score_distribution.png")
plt.close()

# ---------- Correlation Heatmap ----------
correlation_matrix = pdf[features + ["score"]].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    linewidths=0.4
)
plt.title("Correlation Between Features and PE Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_VIS_DIR}/correlation_heatmap.png")
plt.close()


# ------------------------------------------------------------
# 10. FEATURE VECTOR ASSEMBLY (SPARK ML)
# ------------------------------------------------------------

assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)

df_vector = assembler.transform(df)


# ------------------------------------------------------------
# 11. FEATURE SCALING USING MIN-MAX NORMALIZATION
# ------------------------------------------------------------

scaler = MinMaxScaler(
    inputCol="features",
    outputCol="scaled_features"
)

scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)


# ------------------------------------------------------------
# 12. TRAIN–TEST SPLIT
# ------------------------------------------------------------

train_df, test_df = df_scaled.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training records :", train_df.count())
print("Testing records  :", test_df.count())


# ------------------------------------------------------------
# 13. SAVE PROCESSED DATASETS
# ------------------------------------------------------------

train_df.select("scaled_features", "score") \
    .limit(20000) \
    .toPandas() \
    .to_csv(f"{OUTPUT_DATA_DIR}/train_processed.csv", index=False)

test_df.select("scaled_features", "score") \
    .limit(20000) \
    .toPandas() \
    .to_csv(f"{OUTPUT_DATA_DIR}/test_processed.csv", index=False)

print("STEP 2 DATA PREPROCESSING COMPLETED SUCCESSFULLY")

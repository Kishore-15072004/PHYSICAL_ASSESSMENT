import os
import findspark
findspark.init()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

# --------------------------------------------------
# 1. Create visualization folder if not exists
# --------------------------------------------------
VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Start Spark Session
# --------------------------------------------------
spark = SparkSession.builder \
    .appName("PE_Assessment_Step2_Preprocessing") \
    .getOrCreate()

print("Spark version:", spark.version)

# --------------------------------------------------
# 3. Load Dataset
# --------------------------------------------------
df = spark.read.csv(
    "data/pe_assessment_dataset.csv",
    header=True,
    inferSchema=True
)

print("Total records:", df.count())
df.printSchema()

# --------------------------------------------------
# REMOVE DUPLICATE COLUMNS (CRITICAL FIX)
# --------------------------------------------------
unique_cols = []
for c in df.columns:
    if c not in unique_cols:
        unique_cols.append(c)

df = df.select(unique_cols)

print("Columns after removing duplicates:")
print(df.columns)

# --------------------------------------------------
# REMOVE INVALID ZERO ROWS (DOMAIN LOGIC)
# --------------------------------------------------
from pyspark.sql.functions import col, expr

# --------------------------------------------------
# HARD REMOVE INVALID ZERO RECORDS (FINAL FIX)
# --------------------------------------------------
df = df.filter(
    (col("attendance")
     + col("endurance")
     + col("strength")
     + col("flexibility")
     + col("participation")
     + col("skill_speed")
     + col("physical_progress")
     + col("score")) > 0
)

# Force Spark to apply transformations
df = df.persist()
df.count()   # action forces evaluation

print("Final record count after strict zero-row removal:", df.count())

print("Checking for any remaining zero rows:")
df.filter(
    (col("attendance") == 0) &
    (col("endurance") == 0) &
    (col("strength") == 0) &
    (col("flexibility") == 0) &
    (col("participation") == 0) &
    (col("skill_speed") == 0) &
    (col("physical_progress") == 0) &
    (col("score") == 0)
).count()


# --------------------------------------------------
# 4. Missing Value Check
# --------------------------------------------------
df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show()

df = df.dropna()

# --------------------------------------------------
# 5. Convert to Pandas (for visualization only)
# --------------------------------------------------
pdf = df.sample(fraction=0.2, seed=42).toPandas()
# (sampling keeps plots fast & readable)

# --------------------------------------------------
# 6. Feature Distribution Visualization
# --------------------------------------------------
features = [
    "attendance", "endurance", "strength",
    "flexibility", "participation",
    "skill_speed", "physical_progress"
]

plt.figure(figsize=(12, 8))
pdf[features].hist(bins=20, layout=(3, 3))
plt.suptitle("Distribution of Physical Education Features")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/feature_distributions.png")
plt.close()

# --------------------------------------------------
# 7. Score Distribution
# --------------------------------------------------
plt.figure(figsize=(7, 5))
plt.hist(pdf["score"], bins=25)
plt.xlabel("PE Score")
plt.ylabel("Number of Students")
plt.title("Distribution of PE Assessment Scores")
plt.savefig(f"{VIS_DIR}/score_distribution.png")
plt.close()

# --------------------------------------------------
# 8. Correlation Heatmap
# --------------------------------------------------
import seaborn as sns
corr = pdf[features + ["score"]].corr()

plt.figure(figsize=(10, 8))
# 'annot=True' will print the actual correlation numbers in the boxes
# 'cmap' set to coolwarm helps see positive vs negative correlations
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Correlation Between PE Features and Score")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/correlation_heatmap.png")

# --------------------------------------------------
# 9. Feature Vectorization (Spark)
# --------------------------------------------------
assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)

df_vector = assembler.transform(df)

# --------------------------------------------------
# 10. Min-Max Normalization
# --------------------------------------------------
scaler = MinMaxScaler(
    inputCol="features",
    outputCol="scaled_features"
)

scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

# --------------------------------------------------
# 11. Train-Test Split
# --------------------------------------------------
train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)

print("Training records:", train_df.count())
print("Testing records:", test_df.count())

# --------------------------------------------------
# 12. Save Processed Data
# --------------------------------------------------
train_df.select("scaled_features", "score") \
    .toPandas() \
    .to_csv("data/train_processed.csv", index=False)

test_df.select("scaled_features", "score") \
    .toPandas() \
    .to_csv("data/test_processed.csv", index=False)

print("STEP 2 COMPLETED WITH VISUALIZATIONS")

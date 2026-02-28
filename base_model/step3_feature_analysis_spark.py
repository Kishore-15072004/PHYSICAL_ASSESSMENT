import os
import findspark
findspark.init()

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# --------------------------------------------------
# 1. Spark Session
# --------------------------------------------------
spark = SparkSession.builder \
    .appName("PE_Assessment_Step3_Feature_Analysis") \
    .getOrCreate()

print("Spark version:", spark.version)

# --------------------------------------------------
# 2. Load Cleaned Dataset (same as Step 2 output base)
# --------------------------------------------------
df = spark.read.csv(
    "data/pe_assessment_dataset.csv",
    header=True,
    inferSchema=True
)

features = [
    "attendance", "endurance", "strength",
    "flexibility", "participation",
    "skill_speed", "physical_progress"
]

# --------------------------------------------------
# 3. Visualization Directories
# --------------------------------------------------
BASE_DIR = "visualizations/step3"
SCATTER_DIR = f"{BASE_DIR}/feature_vs_score"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(SCATTER_DIR, exist_ok=True)

# --------------------------------------------------
# 4. Statistical Summary (Spark)
# --------------------------------------------------
summary_df = df.select(features + ["score"]).describe()
summary_df.show()

summary_pd = summary_df.toPandas()
summary_pd.to_csv(f"{BASE_DIR}/summary_statistics.csv", index=False)

# --------------------------------------------------
# 5. Convert Sample to Pandas (Visualization only)
# --------------------------------------------------
pdf = df.sample(fraction=0.25, seed=42).toPandas()

# --------------------------------------------------
# 6. Feature vs Score Scatter Plots
# --------------------------------------------------
for feature in features:
    plt.figure(figsize=(6, 4))
    plt.scatter(pdf[feature], pdf["score"], alpha=0.4)
    plt.xlabel(feature)
    plt.ylabel("Final PE Score")
    plt.title(f"{feature} vs PE Score")
    plt.tight_layout()
    plt.savefig(f"{SCATTER_DIR}/{feature}_vs_score.png")
    plt.close()

# --------------------------------------------------
# 7. Correlation Analysis (Numerical Importance)
# --------------------------------------------------
correlation_values = {}

for feature in features:
    corr = df.stat.corr(feature, "score")
    correlation_values[feature] = corr

corr_df = pd.DataFrame.from_dict(
    correlation_values,
    orient="index",
    columns=["Correlation_with_Score"]
).sort_values(by="Correlation_with_Score", ascending=False)

corr_df.to_csv(f"{BASE_DIR}/feature_correlation_values.csv")

# --------------------------------------------------
# 8. Correlation Bar Chart
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.barh(
    corr_df.index,
    corr_df["Correlation_with_Score"]
)
plt.xlabel("Correlation with Final Score")
plt.title("Feature Importance Based on Correlation")
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/correlation_bar.png")
plt.close()

print("STEP 3 COMPLETED SUCCESSFULLY")

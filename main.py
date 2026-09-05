import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA = "students.csv"
MODEL_PATH = "model/student_support_model.joblib"
OUTPUT = "outputs/student_predictions.csv"

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(DATA)

# Create a simple target: students with average academic performance below 65
# are treated as students who may need additional support.
score_cols = ["internal_marks", "assignment_score", "previous_score"]
df["academic_average"] = df[score_cols].mean(axis=1)
df["needs_support"] = (df["academic_average"] < 65).astype(int)

features = ["attendance", "internal_marks", "assignment_score", "previous_score"]
X = df[features]
y = df["needs_support"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced"
    ))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, MODEL_PATH)

df["prediction"] = pipeline.predict(df[features])
df["support_probability"] = pipeline.predict_proba(df[features])[:, 1].round(3)

df["recommendation"] = df["prediction"].map({
    1: "Provide academic support and monitor attendance",
    0: "Continue regular monitoring"
})

df.to_csv(OUTPUT, index=False)

print("AI Student Performance Automation completed.")
print(f"Model saved to: {MODEL_PATH}")
print(f"Predictions saved to: {OUTPUT}")
print("\nStudents needing support:")
print(df.loc[df["prediction"] == 1, ["student_id", "support_probability", "recommendation"]].to_string(index=False))

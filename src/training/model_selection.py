import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv("data/synthetic_telco_churn.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier()
}

# Track Best Model
best_model = None
best_accuracy = 0
best_model_name = ""

mlflow.start_run()

for name, model in models.items():
    print(f"Training {name}...")
    
    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    avg_accuracy = np.mean(scores)

    print(f"{name} Accuracy: {avg_accuracy:.4f}")

    # Log metrics to MLflow
    mlflow.log_metric(f"{name}_accuracy", avg_accuracy)

    # Save best model
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_model = model
        best_model_name = name

# Train best model on full training set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

# Log best model
mlflow.log_metric("best_model_accuracy", final_accuracy)
mlflow.sklearn.log_model(best_model, "best_model")
mlflow.end_run()

print(f"Best Model: {best_model_name} with accuracy {final_accuracy:.4f}")

# Save model for deployment
import joblib
joblib.dump(best_model, "models/best_model.pkl")

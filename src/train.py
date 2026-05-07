import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def train(params, data_path="data/train_phase1.csv", eval_path="data/eval.csv"):
    # 1. Doc du lieu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # --- BONUS 5: Canh bao lech lac du lieu ---
    total_samples = len(df_train)
    dist = df_train["target"].value_counts(normalize=True)
    dist_info = dist.to_dict()
    print(f"\n[Bonus 5] Phan phoi nhan: {dist_info}")
    for label, ratio in dist_info.items():
        if ratio < 0.10:
            print(f"!!! CANH BAO: Lop {label} chi chiem {ratio:.2%}, duoi nguong 10% !!!")

    # 2. Preprocessing
    X_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    X_eval = df_eval.drop("target", axis=1)
    y_eval = df_eval["target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # 3. Chon mo hinh (BONUS 2)
    model_type = params.get("model_type", "random_forest")
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=42
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params.get("max_depth", 3),
            random_state=42
        )
    else:
        clf = LogisticRegression(max_iter=1000)

    # 4. Huan luyen va tracking
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("model_real_type", type(clf).__name__)
        
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_eval_scaled)

        # Tinh toan metrics
        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average="weighted")
        
        # --- BONUS 3: Precision, Recall, Confusion Matrix ---
        precision = precision_score(y_eval, y_pred, average="weighted")
        recall = recall_score(y_eval, y_pred, average="weighted")
        cm = confusion_matrix(y_eval, y_pred)
        report_str = classification_report(y_eval, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Luu output
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # Metrics file (Kem theo phan phoi nhan cho Bonus 5)
        metrics_data = {
            "accuracy": acc, 
            "f1_score": f1,
            "label_distribution": {str(k): float(v) for k, v in dist_info.items()}
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=4)

        # Report file (Bonus 3)
        with open("outputs/report.txt", "w") as f:
            f.write(f"--- Automated Performance Report ---\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(f"Detailed Report:\n{report_str}")

        # Model file
        joblib.dump(clf, "models/model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        mlflow.sklearn.log_model(clf, "model")

        print(f"Huan luyen {model_type} xong. Accuracy: {acc:.4f}")
        return acc

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)

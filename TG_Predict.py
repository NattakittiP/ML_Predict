
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, brier_score_loss,
    roc_curve
)


plt.style.use("default")


if not hasattr(np, "bool"):
    np.bool = bool




DATA_PATH = "WBV_TCR_ICBCB2026_Synthetic_1500_precise_v2 (2).csv"

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())



tg4h_q75 = df["TG4h"].quantile(0.75)
print("\n75th percentile of TG4h:", tg4h_q75)

df["High_TG4h"] = (df["TG4h"] >= tg4h_q75).astype(int)
print("\nLabel distribution (High_TG4h):")
print(df["High_TG4h"].value_counts())
print(df["High_TG4h"].value_counts(normalize=True))


feature_cols = [
    "Sex", "Age", "Hematocrit", "TotalProtein",
    "WBV", "TG0h", "HDL", "LDL", "BMI"
]

X = df[feature_cols].copy()
y = df["High_TG4h"].copy()

cat_features = ["Sex"]
num_features = [c for c in feature_cols if c not in cat_features]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_features),
        ("num", StandardScaler(), num_features),
    ]
)

print("\nNumeric features:", num_features)
print("Categorical features:", cat_features)



models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    )
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model_cv(name, clf, X, y, cv):
    """Leakage-controlled CV evaluation."""
    pipe = Pipeline(steps=[("preprocess", preprocess),
                           ("clf", clf)])
    
    probas = cross_val_predict(
        pipe, X, y,
        cv=cv,
        method="predict_proba"
    )[:, 1]
    preds = (probas >= 0.5).astype(int)
    
    auc = roc_auc_score(y, probas)
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds)
    prec = precision_score(y, preds)
    rec  = recall_score(y, preds)
    brier = brier_score_loss(y, probas)
    
    return {
        "Model": name,
        "AUROC": auc,
        "Accuracy": acc,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "BrierScore": brier
    }

results = []
for name, clf in models.items():
    print(f"\nEvaluating {name} ...")
    res = evaluate_model_cv(name, clf, X, y, skf)
    results.append(res)
    print(res)

results_df = pd.DataFrame(results).sort_values("AUROC", ascending=False)
print("\n=== Cross-validated performance ===")
print(results_df)



best_name = results_df.iloc[0]["Model"]
print("\nBest model by AUROC:", best_name)
best_clf = models[best_name]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


base_pipe = Pipeline(steps=[("preprocess", preprocess),
                            ("clf", best_clf)])


base_pipe.fit(X_train, y_train)

# Calibrate model (isotonic) บน train (cv=5)
calibrated_clf = CalibratedClassifierCV(
    base_pipe, cv=5, method="isotonic"
)
calibrated_clf.fit(X_train, y_train)


probas_test = calibrated_clf.predict_proba(X_test)[:, 1]
preds_test = (probas_test >= 0.5).astype(int)

auc_test = roc_auc_score(y_test, probas_test)
acc_test = accuracy_score(y_test, preds_test)
f1_test  = f1_score(y_test, preds_test)
brier_test = brier_score_loss(y_test, probas_test)

print("\n=== Calibrated model performance on test set ===")
print(f"AUROC   : {auc_test:.3f}")
print(f"Accuracy: {acc_test:.3f}")
print(f"F1      : {f1_test:.3f}")
print(f"Brier   : {brier_test:.3f}")


fpr, tpr, _ = roc_curve(y_test, probas_test)

plt.figure()
plt.plot(fpr, tpr, label=f"{best_name} + calibration (AUROC = {auc_test:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best Calibrated Model (Test Set)")
plt.legend()
plt.tight_layout()
plt.show()


prob_true, prob_pred = calibration_curve(y_test, probas_test, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker="o", linestyle="", label="Calibrated")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Predicted probability")
plt.ylabel("True fraction of positives")
plt.title("Calibration Curve - Best Model (Test Set)")
plt.legend()
plt.tight_layout()
plt.show()



numeric_cols = [
    "Age", "Hematocrit", "TotalProtein", "WBV",
    "TG0h", "TG4h", "TCR", "HDL", "LDL", "BMI"
]

print("\n=== Descriptive statistics by phenotype (mean ± std) ===")
grouped_desc = df.groupby("High_TG4h")[numeric_cols].agg(["mean", "std"])
print(grouped_desc)


corr = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
cax = plt.matshow(corr.values, fignum=0)
plt.colorbar(cax)
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Correlation Matrix of Key Variables")
plt.tight_layout()
plt.show()


plt.figure()
plt.hist(df.loc[df["High_TG4h"] == 0, "TG4h"],
         bins=20, alpha=0.5, label="Normal response")
plt.hist(df.loc[df["High_TG4h"] == 1, "TG4h"],
         bins=20, alpha=0.5, label="High response")
plt.xlabel("TG4h")
plt.ylabel("Count")
plt.title("Distribution of 4-hour TG by Response Phenotype")
plt.legend()
plt.tight_layout()
plt.show()



try:
    import shap
except ImportError:
    
    import shap


shap_pipe = Pipeline(steps=[("preprocess", preprocess),
                            ("clf", LogisticRegression(
                                max_iter=1000,
                                class_weight="balanced",
                                random_state=42
                            ))])

shap_pipe.fit(X, y)


X_trans = shap_pipe.named_steps["preprocess"].transform(X)
clf_lr  = shap_pipe.named_steps["clf"]


ohe = shap_pipe.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(cat_features)
all_feature_names = list(cat_feature_names) + num_features


explainer = shap.LinearExplainer(clf_lr, X_trans)
shap_values = explainer.shap_values(X_trans)


shap.summary_plot(
    shap_values, X_trans,
    feature_names=all_feature_names,
    show=True
)

shap.dependence_plot("TG0h", shap_values, X_trans,feature_names=all_feature_names)


# %% Imports
import json, joblib, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
sys.path.insert(0, "app")
from preprocessing import *

SEED = 42
OUT = "outputs/"

# %% 1. Load
df = pd.read_csv("data/manufacturing_dataset_1000_samples.csv")
print(df.shape); print(df.isna().sum()[lambda s: s > 0]); print("Duplicates:", df.duplicated().sum())

# %% 2. EDA
df[NUM_COLS + [TARGET]].hist(bins=30, figsize=(16, 12)); plt.tight_layout(); plt.savefig(OUT + "eda_hist.png", dpi=120); plt.close()
plt.figure(figsize=(12, 9)); sns.heatmap(df[NUM_COLS + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7}); plt.tight_layout(); plt.savefig(OUT + "eda_corr.png", dpi=120); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c in zip(ax, ["Cycle_Time", "Operator_Experience", "Machine_Age"]):
    sns.scatterplot(data=df, x=c, y=TARGET, ax=a, alpha=.5)
plt.tight_layout(); plt.savefig(OUT + "eda_scatter.png", dpi=120); plt.close()
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, c in zip(ax, CAT_COLS):
    sns.boxplot(data=df, x=c, y=TARGET, ax=a); a.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(OUT + "eda_box_cat.png", dpi=120); plt.close()
print(df[NUM_COLS + [TARGET]].corr()[TARGET].sort_values())
print("Skew of target:", stats.skew(df[TARGET]))

# %% 3. Split FIRST (no leakage), then everything else is learned on train only
X, y = df.drop(columns=TARGET), df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

def build(model, drop=()):
    num = [c for c in NUM_COLS + TIME_COLS if c not in drop]
    cat = [c for c in CAT_COLS if c not in drop]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("clip", IQRClipper()), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(drop="first", handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("time", TimeFeatures()), ("pre", pre), ("model", model)])

# %% 4. Models (all features, as the brief specifies)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="neg_root_mean_squared_error"),
    "Lasso": GridSearchCV(Lasso(max_iter=20000), {"alpha": [0.001, 0.01, 0.1, 1]}, cv=5, scoring="neg_root_mean_squared_error"),
}
rows, fitted = [], {}
for name, m in models.items():
    p = build(m).fit(X_tr, y_tr); fitted[name] = p
    pr = p.predict(X_te)
    cv = cross_val_score(build(m), X_tr, y_tr, cv=5, scoring="r2")
    rows.append(dict(Model=name, R2=r2_score(y_te, pr), MSE=mean_squared_error(y_te, pr), RMSE=mean_squared_error(y_te, pr) ** .5, MAE=mean_absolute_error(y_te, pr), CV_R2_mean=cv.mean(), CV_R2_std=cv.std()))
res = pd.DataFrame(rows).round(4); print(res); res.to_csv(OUT + "model_comparison.csv", index=False)

# %% 5. Ablation: leakage check (Efficiency_Score / Machine_Utilization look like derived outputs)
ab = []
for label, drop in [("All features", ()), ("Without Efficiency_Score & Machine_Utilization", ("Efficiency_Score", "Machine_Utilization")),
                    ("Without all 4 engineered features", ("Efficiency_Score", "Machine_Utilization", "Total_Cycle_Time", "Temperature_Pressure_Ratio"))]:
    p = build(LinearRegression(), drop).fit(X_tr, y_tr); pr = p.predict(X_te)
    ab.append(dict(Setup=label, R2=r2_score(y_te, pr), RMSE=mean_squared_error(y_te, pr) ** .5))
ab = pd.DataFrame(ab).round(4); print(ab); ab.to_csv(OUT + "ablation.csv", index=False)

# %% 6. Final model + residual analysis
# Ridge/Lasso gain is negligible, so keep plain Linear Regression (the brief's model, most interpretable)
# unless a regularised model improves test RMSE by more than 1%.
base = res.set_index("Model")["RMSE"]
best_name = "Linear Regression" if base.min() > 0.99 * base["Linear Regression"] else base.idxmin()
best = fitted[best_name]; pred = best.predict(X_te)
print("Best:", best_name)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].scatter(y_te, pred, alpha=.5); ax[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--"); ax[0].set(title="Predicted vs Actual", xlabel="Actual", ylabel="Predicted")
ax[1].scatter(pred, y_te - pred, alpha=.5); ax[1].axhline(0, color="r"); ax[1].set(title="Residuals vs Predicted", xlabel="Predicted", ylabel="Residual")
sns.histplot(y_te - pred, kde=True, ax=ax[2]); ax[2].set(title="Residual distribution")
plt.tight_layout(); plt.savefig(OUT + "evaluation.png", dpi=120); plt.close()

coef = best[-1].best_estimator_.coef_ if hasattr(best[-1], "best_estimator_") else best[-1].coef_
imp = pd.Series(coef, index=[n.split("__")[-1] for n in best["pre"].get_feature_names_out()]).sort_values(key=abs, ascending=False)
print(imp.head(10)); imp.to_csv(OUT + "coefficients.csv")
imp.head(12)[::-1].plot.barh(figsize=(8, 5), title="Top coefficients (standardised)"); plt.tight_layout(); plt.savefig(OUT + "coefficients.png", dpi=120); plt.close()

# %% 7. Brief's classification metrics via "Underperforming" label (bottom 25% of output, threshold from TRAIN)
thr = y_tr.quantile(0.25)
yt, yp = (y_te < thr).astype(int), (pred < thr).astype(int)
cls = dict(threshold=float(thr), accuracy=accuracy_score(yt, yp), precision=precision_score(yt, yp), recall=recall_score(yt, yp), f1=f1_score(yt, yp))
print(cls)

# %% 8. Per-segment performance
seg = []
for c in ["Shift", "Machine_Type", "Material_Grade"]:
    for v, idx in X_te.groupby(c).groups.items():
        seg.append(dict(Feature=c, Value=v, n=len(idx), RMSE=mean_squared_error(y_te.loc[idx], best.predict(X_te.loc[idx])) ** .5))
pd.DataFrame(seg).round(3).to_csv(OUT + "segment_rmse.csv", index=False)

# %% 9. Save for deployment
joblib.dump(best, "app/model.joblib")
json.dump(dict(best_model=best_name, threshold_underperforming=float(thr), classification=cls, regression=res.set_index("Model").loc[best_name].to_dict()), open(OUT + "summary.json", "w"), indent=2, default=float)
print("saved")

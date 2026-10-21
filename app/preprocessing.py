import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "Parts_Per_Hour"
CAT_COLS = ["Shift", "Machine_Type", "Material_Grade", "Day_of_Week"]
NUM_COLS = [
    "Injection_Temperature", "Injection_Pressure", "Cycle_Time", "Cooling_Time",
    "Material_Viscosity", "Ambient_Temperature", "Machine_Age", "Operator_Experience",
    "Maintenance_Hours", "Temperature_Pressure_Ratio", "Total_Cycle_Time",
    "Efficiency_Score", "Machine_Utilization",
]
TIME_COLS = ["Hour", "Month"]


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Turns Timestamp into Hour and Month; drops Timestamp. Missing Timestamp -> NaN (imputed later)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Timestamp" in X.columns:
            ts = pd.to_datetime(X["Timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
            X["Hour"], X["Month"] = ts.dt.hour, ts.dt.month
            X = X.drop(columns="Timestamp")
        else:
            X["Hour"], X["Month"] = np.nan, np.nan
        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """Learns IQR fences on TRAIN data only and clips (winsorises) values to them."""
    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lo_, self.hi_ = (q1 - self.k * iqr).values, (q3 + self.k * iqr).values
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

from pathlib import Path
from typing import Literal, Optional
import joblib, pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from preprocessing import IQRClipper, TimeFeatures  # noqa: F401  (needed to unpickle the model)

model = joblib.load(Path(__file__).parent / "model.joblib")
UNDERPERFORMING_BELOW = 17.7  # bottom 25% of training output (see outputs/summary.json)

app = FastAPI(title="Injection Molding Output Predictor", version="1.0")


class Machine(BaseModel):
    Injection_Temperature: float = Field(..., ge=150, le=320)
    Injection_Pressure: float = Field(..., ge=50, le=200)
    Cycle_Time: float = Field(..., gt=0, le=120)
    Cooling_Time: float = Field(..., gt=0, le=60)
    Material_Viscosity: Optional[float] = Field(None, gt=0)
    Ambient_Temperature: Optional[float] = None
    Machine_Age: float = Field(..., ge=0, le=40)
    Operator_Experience: Optional[float] = Field(None, ge=0)
    Maintenance_Hours: float = Field(..., ge=0)
    Shift: Literal["Day", "Evening", "Night"]
    Machine_Type: Literal["Type_A", "Type_B", "Type_C"]
    Material_Grade: Literal["Economy", "Standard", "Premium"]
    Day_of_Week: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    Temperature_Pressure_Ratio: Optional[float] = None   # computed if omitted
    Total_Cycle_Time: Optional[float] = None             # computed if omitted
    Efficiency_Score: float = Field(..., ge=0, le=1)
    Machine_Utilization: float = Field(..., ge=0, le=1)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(m: Machine):
    d = m.model_dump()
    if d["Temperature_Pressure_Ratio"] is None:
        d["Temperature_Pressure_Ratio"] = round(d["Injection_Temperature"] / d["Injection_Pressure"], 3)
    if d["Total_Cycle_Time"] is None:
        d["Total_Cycle_Time"] = d["Cycle_Time"] + d["Cooling_Time"]
    pred = float(model.predict(pd.DataFrame([d]))[0])
    pred = max(pred, 0.0)
    return {"predicted_parts_per_hour": round(pred, 2),
            "underperforming": pred < UNDERPERFORMING_BELOW,
            "underperforming_threshold": UNDERPERFORMING_BELOW}
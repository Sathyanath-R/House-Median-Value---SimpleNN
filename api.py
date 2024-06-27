from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import joblib

app = FastAPI()

# Define request body model
class Item(BaseModel):
    # Define request body structure for prediction endpoint
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

# Load preprocessor and model for prediction
preprocessor = joblib.load('preprocessor.pkl')
model = SimpleNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Prediction endpoint
@app.post("/predict")
async def predict(item: Item):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([item.dict()])
    
    # Preprocess input data
    X_processed = preprocessor.transform(input_data)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    
    # Perform prediction using the trained model
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor).item()
    
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
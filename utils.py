# utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from typing import Tuple, Any
import joblib

def load_data(file_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def preprocess_data(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[np.ndarray, pd.Series, ColumnTransformer]:
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(include=['number']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    y_train_processed = y_train

    return X_train_processed, y_train_processed, preprocessor

class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model: nn.Module, train_loader: data.DataLoader, lr: float, epochs: int) -> None:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def evaluate_model(model: nn.Module, X_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        mse = nn.MSELoss()(predictions, y_test_tensor).item()
        r2 = r2_score(y_test_tensor.numpy(), predictions.numpy())
    print(f'Test Loss (MSE): {mse}')
    print(f'R^2 Score: {r2}')

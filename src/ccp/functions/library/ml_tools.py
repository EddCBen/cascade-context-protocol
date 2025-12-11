from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.ccp.functions.utils import expose_as_ccp_tool

@expose_as_ccp_tool
def run_anomaly_detection(data_vector: List[float], contamination: float = 0.1) -> Dict[str, Any]:
    """
    Detects anomalies in a list of float values using Isolation Forest.
    
    Args:
        data_vector (List[float]): The input time-series or sequence of data points.
        contamination (float): The expected proportion of outliers in the data set (default: 0.1).
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'anomalies': List of indices where anomalies were detected.
            - 'scores': List of anomaly scores for each data point (lower is more anomalous).
            - 'is_anomaly': List of booleans corresponding to each data point.
    """
    if not data_vector:
        return {"error": "Empty data vector provided"}
        
    # Preprocessing
    X = np.array(data_vector).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model Inference
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(X_scaled)
    # -1 for outliers, 1 for inliers
    
    scores = clf.decision_function(X_scaled)
    
    anomalies_indices = [i for i, x in enumerate(predictions) if x == -1]
    is_anomaly = [bool(x == -1) for x in predictions]
    
    return {
        "anomalies_indices": anomalies_indices,
        "scores": scores.tolist(),
        "is_anomaly": is_anomaly,
        "total_anomalies": len(anomalies_indices)
    }

@expose_as_ccp_tool
def forecast_series(history: List[float], horizon: int = 3) -> Dict[str, Any]:
    """
    Forecasts future values of a time series using a simple Linear Regression model
    on auto-regressive features.
    
    Args:
        history (List[float]): Historical data points (ordered chronologically).
        horizon (int): Number of future steps to predict (default: 3).
        
    Returns:
        Dict[str, Any]:
            - 'forecast': List of predicted future values.
            - 'method': "LinearRegression"
    """
    if len(history) < 5:
        return {"error": "Insufficient history for forecasting (need at least 5 points)"}
    
    # Simple formatting for regression: y = f(t)
    # We'll map time index to value for simple trend extrapolation
    # A robust LSTM is requested "or LinearRegression for speed". We chose LR for speed/reliability in this context.
    
    X = np.arange(len(history)).reshape(-1, 1)
    y = np.array(history)
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    # Predict future
    last_idx = len(history)
    future_X = np.arange(last_idx, last_idx + horizon).reshape(-1, 1)
    future_X_scaled = scaler_X.transform(future_X)
    
    pred_scaled = model.predict(future_X_scaled)
    forecast = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    return {
        "forecast": forecast.tolist(),
        "method": "LinearRegression (Trend Extrapolation)"
    }

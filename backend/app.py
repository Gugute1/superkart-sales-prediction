
# Import necessary libraries
import io
import os
import joblib
import numpy as np
import pandas as pd
import json
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

superkart_api = FastAPI(title="SuperKart Store Sales Prediction API")

# 1. Load the unified model and preprocessing bundle once at startup
saved_model_name = "superkart_model_and_preprocessing_bundle.joblib"
bundle_model_path = (
    f"backend_files/{saved_model_name}"
    if os.path.exists(f"backend_files/{saved_model_name}")
    else saved_model_name
)
loaded_bundle = joblib.load(bundle_model_path)
model = loaded_bundle["model"]

# Define a route for the home page
#
# Benefits of adding a root route (get / route):
#     1) Cloud Hosting Health Checks: Platforms like Hugging Face Spaces, Render, and Google Cloud Run frequently send GET / requests to verify 
#                                     that the model container booted properly. A 200 OK response tells the platform your API is healthy and ready 
#                                     to receive traffic.
#     2) Instant Browser Verification: Opening the live backend URL in a web browser immediately confirms the server is online instead of displaying 
#                                     a confusing {"detail": "Not Found"} message.
#     3) API Navigation Aid: Returning a clean JSON status message with links to the /docs, /v1/predict, and /v1/predict_batch endpoints makes the 
#                                     service self-documenting.

@superkart_api.get("/")
def home():
    return {
        "status": "online",
        "message": "SuperKart Sales Prediction API is running.",
        "docs_url": "/docs",
        "endpoints": {
            "health_check": "/health",
            "single_prediction": "/v1/predict",
            "batch_prediction_JSON_list": "/v1/predict_json_list",
            "batch_prediction_CSV_upload": "/v1/predict_batch"
        }
    }


# /health endpoint verifies if the .joblib model bundle loaded cleanly into memory
@superkart_api.get("/health")
def health_check():
    # Verify that the model artifact is loaded and ready in memory
    if loaded_bundle is None or "model" not in loaded_bundle:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Model artifact failed to load."  # A model artifact is the saved, packaged file (or bundle of files) created 
                                                                          # after training a machine learning model that contains everything needed to 
                                                                          # make predictions in production without retraining.
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(loaded_bundle["model"]).__name__
    }


# A class is a blueprint or template for creating structured objects. It defines what attributes (data) an object must hold and 
# what behaviors (methods) it can perform.
# In FastAPI code, class ProductInput(BaseModel): defines a strict data validation contract (schema) for incoming API requests 
# using Pydantic's BaseModel.

# 2. Define the expected raw JSON input schema
class ProductInput(BaseModel):
    Product_Weight: float
    Product_Sugar_Content: str
    Product_Allocated_Area: float
    Product_MRP: float
    Store_Size: str
    Store_Location_City_Type: str
    Store_Type: str
    Product_ID_Char: str
    Store_Age_Years: int
    Product_Type_Category: str

# 3. Helper function replicating the preprocessing pipeline
def preprocess_incoming_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # A. Normalize all text/object columns
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        df[col] = df[col].astype("string").str.lower().str.strip()

    # B. Map incoming API field names to training feature names (fixed exact casing)
    df["store_age_in_years"] = df["Store_Age_Years"].copy()

    # C. Clean sugar content before applying ColumnTransformer
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace(loaded_bundle["sugar_mapping"])

    # D. Convert Pandas nullable 'string' dtype back to standard 'object' for Scikit-Learn compatibility
    string_cols = df.select_dtypes(include=["string"]).columns
    for col in string_cols:
        df[col] = df[col].astype("object")

    # E. Map target-encoding group means with safe global fallbacks
    df["Product_Type_Avg_MRP"] = df["Product_Type_Category"].map(loaded_bundle["mrp_means_map"])
    df = df.merge(loaded_bundle["area_means_df"], on=["Store_Type", "Product_Type_Category"], how="left")

    # F. Advanced Feature Engineering
    df["visibility_per_dollar"] = 100 * df["Product_Allocated_Area"] / (df["Product_MRP"] + 1e-5)
    df["MRP_vs_category_mean"] = df["Product_MRP"] / (df["Product_Type_Avg_MRP"] + 1e-5)
    df["weight_allocation_density"] = df["Product_Weight"] * df["Product_Allocated_Area"]
    df["category_area_domination"] = df["Product_Allocated_Area"] / (df["Store_Product_Avg_Area"] + 1e-5)

    engineered_cols = ["visibility_per_dollar", "MRP_vs_category_mean", "weight_allocation_density", "category_area_domination"]
    for col in engineered_cols:
        if col in df.columns and df[col].dtype != "object":
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # G. Run the fitted ColumnTransformer
    processed_array = loaded_bundle["preprocessor"].transform(df)
    feature_names = loaded_bundle["preprocessor"].get_feature_names_out()
    processed_df = pd.DataFrame(processed_array, columns=feature_names)
    processed_df.columns = processed_df.columns.str.replace(" ", "_")

    # H. Drop baseline dummy variables and temporary lookup columns
    processed_df = processed_df.drop(columns=loaded_bundle["vars_to_drop"], errors="ignore")

    # I. Reindex to enforce the exact column set and order expected by CatBoost
    final_df = processed_df.reindex(columns=loaded_bundle["final_feature_columns"], fill_value=0)

    return final_df

# 4. Endpoint #1: Single Item Prediction
@superkart_api.post("/v1/predict")
def predict_single(item: ProductInput):
    raw_df = pd.DataFrame([item.model_dump()])
    X_ready = preprocess_incoming_data(raw_df)

    log_pred = model.predict(X_ready)
    real_sales_pred = float(np.clip(np.expm1(log_pred[0]), 0, None))

    return {
        "status": "success",
        "predicted_sales_total": round(real_sales_pred, 2),
        "currency": "USD"
    }


# 5. Endpoint #2: Batch Prediction (JSON List of Items)
@superkart_api.post("/v1/predict_json_list")
def predict_batch(items: list[ProductInput]):
    if not items:
        raise HTTPException(status_code=400, detail="Input list cannot be empty.")

    raw_df = pd.DataFrame([item.model_dump() for item in items])
    X_ready = preprocess_incoming_data(raw_df)

    log_preds = model.predict(X_ready)
    real_sales_preds = np.clip(np.expm1(log_preds), 0, None)

    predictions_output = [
        {
            "item_index": idx,
            "predicted_sales_total": round(float(pred), 2)
        }
        for idx, pred in enumerate(real_sales_preds)
    ]

    return {
        "status": "success",
        "count": len(predictions_output),
        "currency": "USD",
        "predictions": predictions_output
    }


# 6. Endpoint #3: Batch Prediction via CSV Upload
@superkart_api.post("/v1/predict_batch")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    contents = await file.read()
    raw_df = pd.read_csv(io.BytesIO(contents))

    if raw_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    # Verify all required schema columns exist before preprocessing
    required_cols = list(ProductInput.model_fields.keys())
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing required CSV columns: {missing_cols}")

    X_ready = preprocess_incoming_data(raw_df[required_cols])
    log_preds = model.predict(X_ready)
    real_sales_preds = np.clip(np.expm1(log_preds), 0, None)

    results = [
        {
            "row_index": idx,
            "predicted_sales_total": round(float(pred), 2)
        }
        for idx, pred in enumerate(real_sales_preds)
    ]

    return {
        "status": "success",
        "count": len(results),
        "currency": "USD",
        "predictions": results
    }


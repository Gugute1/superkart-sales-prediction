# SuperKart Store Sales Prediction (End-to-End ML Deployment)

End-to-end machine learning pipeline and containerized microservices architecture for forecasting product-level store sales across SuperKart outlets.

## Repository Structure
```text
superkart-sales-prediction/
├── backend/
│   ├── app.py                                           # FastAPI inference service
│   ├── superkart_model_and_preprocessing_bundle.joblib  # Serialized CatBoost & preprocessing artifact
│   ├── requirements.txt                                 # Backend Python dependencies
│   └── Dockerfile                                       # Container build instructions (Port 8000)
├── frontend/
│   ├── app_ui.py                                        # Streamlit interactive UI
│   ├── requirements.txt                                 # Frontend Python dependencies
│   └── Dockerfile                                       # Container build instructions (Port 8501)
└── README.md
```

## FastAPI Endpoints (`/v1`)

| Method | Endpoint                | Description                                               |
| :----- | :---------------------- | :-------------------------------------------------------- |
| `GET`  | `/`                     | Root liveness check and endpoint directory                |
| `GET`  | `/health`               | Model readiness check (`CatBoostRegressor` loaded status) |
| `POST` | `/v1/predict`           | Single-item JSON prediction                               |
| `POST` | `/v1/predict_json_list` | Batch JSON list prediction                                |
| `POST` | `/v1/predict_batch`     | Batch `.csv` file upload prediction                       |

## Local Docker Execution
```bash
# 1. Build & run FastAPI backend
docker build -t superkart-backend ./backend
docker run -d -p 8000:8000 superkart-backend

# 2. Build & run Streamlit frontend
docker build -t superkart-frontend ./frontend
docker run -d -p 8501:8501 -e BACKEND_URL="http://localhost:8000" superkart-frontend
```

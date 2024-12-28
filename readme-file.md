# Customer Churn Prediction Pipeline

A production-ready machine learning pipeline for predicting customer churn using FastAPI and Prometheus monitoring.

## Features

- Real-time churn prediction via REST API
- Automated weekly model retraining
- Data validation and quality monitoring
- Feature engineering pipeline
- Prometheus metrics integration
- Comprehensive logging
- Production-ready error handling

## Project Structure

```
customer-churn-prediction/
├── data/                  # Data storage directory
├── models/               # Trained models directory
├── metrics/             # Metrics storage directory
├── src/                 # Source code
│   └── pipeline.py      # Main pipeline implementation
├── requirements.txt     # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Chukwuemekaokafor77/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the pipeline:
```bash
python src/pipeline.py
```

This will:
- Start a Prometheus metrics server on port 8000
- Run initial model training
- Start the FastAPI server on port 8001
- Schedule weekly pipeline runs

2. Access the API documentation at:
```
http://localhost:8001/docs
```

## Monitoring

Prometheus metrics are available at:
```
http://localhost:8000/metrics
```

Available metrics:
- prediction_latency_seconds: Time spent processing predictions
- data_quality_score: Current data quality score
- model_performance_auc: Current model AUC-ROC score
- predictions_total: Total number of predictions made

## API Endpoints

### Predict Churn

```
POST /predict
```

Request body:
```json
{
    "customer_id": "CUST_123",
    "signup_date": "2023-01-01",
    "last_active": "2023-12-01",
    "total_spend": 1500.0,
    "num_support_tickets": 3,
    "total_logins": 45,
    "subscription_type": "premium",
    "country": "USA"
}
```

Response:
```json
{
    "customer_id": "CUST_123",
    "churn_probability": 0.75,
    "prediction_timestamp": "2024-12-27T10:00:00"
}
```

## License

MIT License

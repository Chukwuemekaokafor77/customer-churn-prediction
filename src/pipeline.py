import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import joblib
from datetime import datetime, timedelta
import logging
import yaml
from typing import Dict, List, Tuple, Optional
import schedule
import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from pandera import DataFrameSchema, Column, Check
import prometheus_client as prom

# Initialize FastAPI
app = FastAPI()

# Initialize Prometheus metrics
PREDICTION_LATENCY = prom.Summary('prediction_latency_seconds', 'Time spent processing prediction')
DATA_QUALITY_SCORE = prom.Gauge('data_quality_score', 'Current data quality score')
MODEL_PERFORMANCE = prom.Gauge('model_performance_auc', 'Current model AUC-ROC score')
PREDICTIONS_TOTAL = prom.Counter('predictions_total', 'Total number of predictions made')


class DataGenerator:
    def __init__(self, num_customers: int = 1000):
        self.num_customers = num_customers
        self.logger = logging.getLogger(__name__)

    def generate_batch(self) -> pd.DataFrame:
        try:
            current_date = datetime.now()
            data = {
                'customer_id': [f'CUST_{i}' for i in range(self.num_customers)],
                'signup_date': [
                    (current_date - timedelta(days=np.random.randint(30, 365))).strftime('%Y-%m-%d')
                    for _ in range(self.num_customers)
                ],
                'last_active': [
                    (current_date - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d')
                    for _ in range(self.num_customers)
                ],
                'total_spend': np.random.uniform(100, 5000, self.num_customers),
                'num_support_tickets': np.random.randint(0, 10, self.num_customers, dtype=np.int64),
                'total_logins': np.random.randint(1, 200, self.num_customers, dtype=np.int64),
                'subscription_type': np.random.choice(
                    ['basic', 'premium', 'enterprise'], self.num_customers
                ),
                'country': np.random.choice(
                    ['USA', 'UK', 'Canada', 'Australia'], self.num_customers
                )
            }
            df = pd.DataFrame(data)
            df['churned'] = (
                (df['total_logins'] < 50) & 
                (df['num_support_tickets'] > 5) & 
                (df['total_spend'] < 1000)
            ).astype(int)
            self.logger.info(f"Generated {len(df)} rows of synthetic data")
            return df
        except Exception as e:
            self.logger.error(f"Error in data generation: {str(e)}")
            raise

class DataIngestion:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_batch(self, df: pd.DataFrame) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_dir}/batch_{timestamp}.csv"
        df.to_csv(filename, index=False)
        self.logger.info(f"Saved batch data to {filename}")
        return filename

    def load_latest_batch(self) -> pd.DataFrame:
        try:
            files = os.listdir(self.data_dir)
            if not files:
                raise FileNotFoundError("No data files found")
            latest_file = max(files)
            df = pd.read_csv(f"{self.data_dir}/{latest_file}")
            self.logger.info(f"Loaded latest batch from {latest_file}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading latest batch: {str(e)}")
            raise

class FeaturePipeline:
    def __init__(self, config_path: str = 'config.yaml'):
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            features = df.copy()
            features['customer_lifetime'] = (
                pd.to_datetime(features['last_active']) - pd.to_datetime(features['signup_date'])
            ).dt.days
            features['spend_per_day'] = features['total_spend'] / features['customer_lifetime']
            features['logins_per_day'] = features['total_logins'] / features['customer_lifetime']
            features['tickets_per_day'] = features['num_support_tickets'] / features['customer_lifetime']
            features = pd.get_dummies(
                features, columns=['subscription_type', 'country'], prefix=['sub', 'country']
            )
            feature_cols = [col for col in features.columns if col not in ['customer_id', 'signup_date', 'last_active', 'churned']]
            features[feature_cols] = self.scaler.fit_transform(features[feature_cols])
            return features[feature_cols]
        except Exception as e:
            self.logger.error(f"Error in feature processing: {str(e)}")
            raise

class ModelManager:
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_predictions(self, predictions: np.ndarray, customer_ids: List[str]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_df = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_probability': predictions,
            'prediction_time': timestamp
        })
        filename = f"{self.models_dir}/predictions_{timestamp}.csv"
        predictions_df.to_csv(filename, index=False)
        self.logger.info(f"Saved predictions to {filename}")

class InferencePipeline:
    def __init__(self, model_path: str, feature_pipeline: FeaturePipeline):
        self.model = joblib.load(model_path)
        self.feature_pipeline = feature_pipeline
        self.logger = logging.getLogger(__name__)

    def predict(self, raw_data: pd.DataFrame) -> np.ndarray:
        try:
            features = self.feature_pipeline.process_raw_data(raw_data)
            predictions = self.model.predict_proba(features)[:, 1]
            self.logger.info(f"Generated predictions for {len(raw_data)} instances")
            return predictions
        except Exception as e:
            self.logger.error(f"Error in prediction generation: {str(e)}")
            raise

class ModelTrainer:
    def __init__(self, config_path: str = 'config.yaml'):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.logger = logging.getLogger(__name__)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            self.model.fit(X, y)
            model_path = f'models/latest_model.joblib'
            joblib.dump(self.model, model_path)
            self.logger.info(f"Model trained and saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

class DataValidator:
    def __init__(self):
        self.schema = DataFrameSchema({
            'customer_id': Column(str),
            'signup_date': Column(str, Check(lambda x: pd.to_datetime(x, errors='coerce').notna().all())),
            'last_active': Column(str, Check(lambda x: pd.to_datetime(x, errors='coerce').notna().all())),
            'total_spend': Column(float, Check(lambda x: (x >= 0) & (x <= 1000000))),
            'num_support_tickets': Column(int, Check(lambda x: (x >= 0) & (x <= 1000))),
            'total_logins': Column(int, Check(lambda x: (x >= 0) & (x <= 10000))),
            'subscription_type': Column(str, Check(lambda x: x.isin(['basic', 'premium', 'enterprise']))),
            'country': Column(str)
        })
        self.logger = logging.getLogger(__name__)

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        try:
            # Convert integer columns to int64 before validation
            df = df.copy()
            integer_columns = ['num_support_tickets', 'total_logins']
            for col in integer_columns:
                df[col] = df[col].astype(np.int64)
            
            self.schema.validate(df)
            metrics = {
                'missing_values_pct': df.isnull().mean().mean() * 100,
                'duplicate_rows_pct': (df.duplicated().sum() / len(df)) * 100,
                'data_points': len(df),
                'timestamp': datetime.now().isoformat()
            }
            quality_score = 100 - metrics['missing_values_pct'] - metrics['duplicate_rows_pct']
            DATA_QUALITY_SCORE.set(quality_score)
            return True, metrics
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False, {'error': str(e)}

class MetricsTracker:
    def __init__(self, metrics_dir: str = 'metrics'):
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred),
            'avg_precision': average_precision_score(y_true, y_pred),
            'timestamp': datetime.now().isoformat()
        }
        MODEL_PERFORMANCE.set(metrics['auc_roc'])
        return metrics

    def save_metrics(self, metrics: Dict) -> None:
        filename = f"{self.metrics_dir}/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(metrics, f)
        self.logger.info(f"Saved metrics to {filename}")

class PredictionRequest(BaseModel):
    customer_id: str
    signup_date: str
    last_active: str
    total_spend: float
    num_support_tickets: int
    total_logins: int
    subscription_type: str
    country: str

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction_timestamp: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    try:
        with PREDICTION_LATENCY.time():
            # Convert request to DataFrame
            data = pd.DataFrame([request.dict()])
            
            # Load feature pipeline and model
            feature_pipeline = FeaturePipeline()
            inference = InferencePipeline('models/latest_model.joblib', feature_pipeline)
            
            # Generate prediction
            prediction = inference.predict(data)[0]
            PREDICTIONS_TOTAL.inc()
            
            return PredictionResponse(
                customer_id=request.customer_id,
                churn_probability=float(prediction),
                prediction_timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_weekly_pipeline():
    try:
        logging.info("Starting weekly pipeline run")
        data_generator = DataGenerator()
        data_ingestion = DataIngestion()
        feature_pipeline = FeaturePipeline()
        model_manager = ModelManager()
        validator = DataValidator()
        metrics_tracker = MetricsTracker()

        new_data = data_generator.generate_batch()
        is_valid, quality_metrics = validator.validate_data(new_data)
        if not is_valid:
            raise ValueError(f"Data validation failed: {quality_metrics['error']}")

        data_ingestion.save_batch(new_data)
        try:
            inference = InferencePipeline('models/latest_model.joblib', feature_pipeline)
        except FileNotFoundError:
            features = feature_pipeline.process_raw_data(new_data)
            trainer = ModelTrainer()
            trainer.train(features, new_data['churned'])
            inference = InferencePipeline('models/latest_model.joblib', feature_pipeline)

        predictions = inference.predict(new_data)
        metrics = metrics_tracker.calculate_metrics(new_data['churned'], predictions)
        metrics_tracker.save_metrics(metrics)
        model_manager.save_predictions(predictions, new_data['customer_id'])
        logging.info("Completed weekly pipeline run")
    except Exception as e:
        logging.error(f"Pipeline run failed: {str(e)}")
        raise

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    prom.start_http_server(8000)
    schedule.every().monday.at("00:00").do(run_weekly_pipeline)
    
    # Run initial pipeline
    run_weekly_pipeline()
    
    # Start FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()
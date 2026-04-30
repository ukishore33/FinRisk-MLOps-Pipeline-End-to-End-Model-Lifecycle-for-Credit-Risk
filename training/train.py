"""MLflow-instrumented XGBoost training with full experiment tracking."""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LoanDefaultTrainer:
    """Training pipeline with MLflow integration."""
    
    def __init__(self, mlflow_uri: str = "http://localhost:5000"):
        """Initialize trainer."""
        self.mlflow_uri = mlflow_uri
        self.experiment_name = "loan-default-xgboost"
        self.model_name = "loan-default-xgboost"
        self.model = None
        self.feature_names = None
        self.label_encoders = {}
        
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(self.experiment_name)
    
    def load_and_preprocess(self, data_path: str) -> tuple:
        """Load data and preprocess."""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        logger.info(f"Dataset shape: {df.shape}")
        
        # Drop loan_id (not needed for training)
        df = df.drop(['loan_id'], axis=1)
        
        # Separate target
        y = df['loan_default']
        X = df.drop(['loan_default'], axis=1)
        
        # Encode categorical features
        categorical_cols = [
            'employment_type', 'education', 'residence_type', 'loan_purpose'
        ]
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        logger.info(f"Features: {list(X.columns)}")
        self.feature_names = list(X.columns)
        
        return X, y
    
    def train_and_log(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        hyperparams: dict = None
    ):
        """Train model and log to MLflow."""
        
        if hyperparams is None:
            hyperparams = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'tree_method': 'hist'
            }
        
        with mlflow.start_run(run_name="xgboost-training"):
            # Log hyperparameters
            mlflow.log_params(hyperparams)
            
            # Train model
            logger.info("Training XGBoost model...")
            self.model = xgb.XGBClassifier(**hyperparams)
            
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Predictions on all sets
            y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]
            y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            y_train_pred = self.model.predict(X_train)
            y_val_pred = self.model.predict(X_val)
            y_test_pred = self.model.predict(X_test)
            
            # Compute metrics
            train_auc = roc_auc_score(y_train, y_train_pred_proba)
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            test_auc = roc_auc_score(y_test, y_test_pred_proba)
            
            # Gini coefficient (2 * AUC - 1)
            test_gini = 2 * test_auc - 1
            
            # KS statistic
            fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
            ks_stat = np.max(np.abs(tpr - fpr))
            
            # Precision, Recall, F1
            precision = np.sum((y_test_pred == 1) & (y_test == 1)) / (np.sum(y_test_pred == 1) + 1e-8)
            recall = np.sum((y_test_pred == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-8)
            test_f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Log metrics
            metrics = {
                'train_auc': train_auc,
                'val_auc': val_auc,
                'test_auc': test_auc,
                'test_gini': test_gini,
                'test_ks': ks_stat,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': test_f1,
                'test_default_rate': y_test.mean(),
            }
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
                logger.info(f"{key}: {value:.4f}")
            
            # Log model
            mlflow.xgboost.log_model(
                self.model,
                "model",
                registered_model_name=self.model_name
            )
            
            # Log feature importance
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Features:")
            logger.info(importance_df.head(10).to_string())
            
            # Log feature importance as artifact
            importance_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
            logger.info(f"✓ Model logged to MLflow")
            
            return {
                'model': self.model,
                'metrics': metrics,
                'feature_importance': importance_df
            }


def main():
    """Main training function."""
    # Configuration
    DATA_PATH = os.getenv("DATA_PATH", "data/loan_data.csv")
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Initialize trainer
    trainer = LoanDefaultTrainer(mlflow_uri=MLFLOW_URI)
    
    # Load and preprocess data
    X, y = trainer.load_and_preprocess(DATA_PATH)
    
    # Split data: 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Default rate - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")
    
    # Train with MLflow logging
    results = trainer.train_and_log(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )
    
    logger.info("\n✓ Training complete!")
    logger.info(f"Test Gini: {results['metrics']['test_gini']:.4f}")
    logger.info(f"Test KS: {results['metrics']['test_ks']:.4f}")


if __name__ == "__main__":
    main()
    
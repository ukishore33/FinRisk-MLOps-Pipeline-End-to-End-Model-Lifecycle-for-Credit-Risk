"""Load model from MLflow registry or local joblib file."""
import os
import joblib
import logging
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient


logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage model lifecycle."""
    
    def __init__(self, model_name: str = "loan-default-xgboost"):
        self.model_name = model_name
        self.model = None
        self.model_version = None
        self.preprocessor = None
        self.feature_names = None
        self.mlflow_client = None
        
        # Initialize MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        self.mlflow_client = MlflowClient(tracking_uri=mlflow_uri)
    
    def load_production_model(self) -> bool:
        """Load model from MLflow Production stage."""
        try:
            logger.info(f"Loading model '{self.model_name}' from MLflow Production stage")
            
            # Get latest production version
            versions = self.mlflow_client.get_latest_versions(
                self.model_name, 
                stages=["Production"]
            )
            
            if not versions:
                logger.warning(f"No Production version found for {self.model_name}")
                return self._load_fallback_model()
            
            prod_version = versions[0]
            self.model_version = prod_version.version
            
            # Load model artifact
            model_uri = f"models:/{self.model_name}/Production"
            self.model = mlflow.xgboost.load_model(model_uri)
            
            # Load preprocessor
            preprocessor_uri = f"models:/{self.model_name}/Production"
            try:
                self.preprocessor = mlflow.sklearn.load_model(
                    f"{preprocessor_uri}:preprocessor"
                )
            except Exception as e:
                logger.warning(f"Could not load preprocessor from MLflow: {e}")
            
            logger.info(f"✓ Loaded {self.model_name} version {self.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load from MLflow: {e}")
            return self._load_fallback_model()
    
    def _load_fallback_model(self) -> bool:
        """Fallback: load local joblib model."""
        try:
            model_path = os.getenv(
                "LOCAL_MODEL_PATH", 
                "./models/xgboost_model.joblib"
            )
            
            if not os.path.exists(model_path):
                logger.error(f"Local model not found at {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            self.model_version = "local"
            
            preprocessor_path = os.getenv(
                "LOCAL_PREPROCESSOR_PATH",
                "./models/preprocessor.joblib"
            )
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
            
            logger.info("✓ Loaded fallback local model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            return False
    
    def get_feature_names(self) -> list:
        """Get expected feature names from model."""
        if hasattr(self.model, 'get_booster'):
            # XGBoost model
            return self.model.get_booster().feature_names
        return self.feature_names
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Singleton loader
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create singleton model loader."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.load_production_model()
    return _model_loader
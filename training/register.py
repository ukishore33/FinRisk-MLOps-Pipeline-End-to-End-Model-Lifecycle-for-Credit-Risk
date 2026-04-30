"""Promote model to Production in MLflow registry if performance is good."""
import mlflow
from mlflow.tracking import MlflowClient
import logging


logger = logging.getLogger(__name__)


def promote_to_production(
    model_name: str,
    min_gini: float = 0.40,
    mlflow_uri: str = "http://localhost:5000"
):
    """
    Promote latest model version to Production if Gini > threshold.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)
    
    try:
        # Get latest version
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            logger.error(f"No versions found for {model_name}")
            return False
        
        latest_version = max(versions, key=lambda x: int(x.version))
        run_id = latest_version.run_id
        
        # Get metrics from run
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        
        test_gini = metrics.get('test_gini', 0)
        
        logger.info(f"Latest version {latest_version.version} has Gini: {test_gini:.3f}")
        
        if test_gini >= min_gini:
            # Archive current Production
            try:
                current_prod = client.get_latest_versions(model_name, stages=["Production"])
                if current_prod:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=current_prod[0].version,
                        stage="Archived"
                    )
                    logger.info(f"Archived version {current_prod[0].version}")
            except Exception as e:
                logger.warning(f"Could not archive current production: {e}")
            
            # Promote new version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Production"
            )
            
            logger.info(
                f"✓ Promoted version {latest_version.version} to Production "
                f"(Gini: {test_gini:.3f})"
            )
            return True
        
        else:
            logger.warning(
                f"✗ Model not promoted: Gini {test_gini:.3f} < threshold {min_gini}"
            )
            return False
    
    except Exception as e:
        logger.error(f"Promotion failed: {e}")
        return False


if __name__ == "__main__":
    promote_to_production("loan-default-xgboost", min_gini=0.40)
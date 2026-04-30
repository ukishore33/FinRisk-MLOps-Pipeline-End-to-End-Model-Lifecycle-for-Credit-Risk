"""Evidently AI data drift & model monitoring reports."""
import pandas as pd
import logging
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable, ClassificationDriftMetrics, DataQualityTable
)
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues, TestNumberOfDuplicatedRows


logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data and prediction drift using Evidently AI."""
    
    def __init__(self):
        self.reference_data = None
        self.reference_predictions = None
    
    def set_reference(
        self,
        X_ref: pd.DataFrame,
        y_ref: pd.Series
    ):
        """Set reference dataset for comparison."""
        self.reference_data = X_ref
        self.reference_predictions = y_ref
        logger.info(f"Reference set with {len(X_ref)} samples")
    
    def detect_data_drift(
        self,
        X_current: pd.DataFrame,
        y_current: pd.Series,
        output_path: str = "drift_report.html"
    ) -> dict:
        """Detect data drift between reference and current data."""
        
        logger.info(f"Detecting data drift ({len(X_current)} current samples)...")
        
        # Create data drift report
        report = Report(metrics=[
            DataDriftTable(),
            DataQualityTable(),
        ])
        
        report.run(
            reference_data=self.reference_data.assign(target=self.reference_predictions),
            current_data=X_current.assign(target=y_current)
        )
        
        # Save HTML report
        report.save_html(output_path)
        logger.info(f"Drift report saved to {output_path}")
        
        # Extract drift summary
        drift_results = report.as_dict()
        
        return {
            'report_path': output_path,
            'drift_detected': self._check_drift_threshold(drift_results),
            'summary': drift_results
        }
    
    def detect_prediction_drift(
        self,
        y_ref: pd.Series,
        y_current: pd.Series,
        y_pred_proba_ref: pd.Series,
        y_pred_proba_current: pd.Series,
        output_path: str = "prediction_drift_report.html"
    ) -> dict:
        """Detect prediction drift (target distribution shift)."""
        
        logger.info("Detecting prediction drift...")
        
        report = Report(metrics=[
            ClassificationDriftMetrics(),
        ])
        
        report.run(
            reference_data=pd.DataFrame({
                'target': y_ref,
                'prediction': y_pred_proba_ref
            }),
            current_data=pd.DataFrame({
                'target': y_current,
                'prediction': y_pred_proba_current
            })
        )
        
        report.save_html(output_path)
        logger.info(f"Prediction drift report saved to {output_path}")
        
        return {
            'report_path': output_path,
            'summary': report.as_dict()
        }
    
    def run_quality_tests(self, X_current: pd.DataFrame) -> dict:
        """Run data quality tests."""
        
        suite = TestSuite(tests=[
            TestNumberOfMissingValues(),
            TestNumberOfDuplicatedRows(),
        ])
        
        suite.run(reference_data=self.reference_data, current_data=X_current)
        
        return {
            'passed': suite.passed,
            'failed': suite.failed,
            'summary': suite.as_dict()
        }
    
    @staticmethod
    def _check_drift_threshold(drift_results: dict, threshold: float = 0.3) -> bool:
        """Check if drift exceeds threshold."""
        # Simplified: in production, parse drift_results properly
        return False  # TODO: Implement threshold logic


def generate_monitoring_report(
    X_ref: pd.DataFrame,
    y_ref: pd.Series,
    X_current: pd.DataFrame,
    y_current: pd.Series,
    y_pred_proba_ref: pd.Series = None,
    y_pred_proba_current: pd.Series = None,
    output_dir: str = "./monitoring_reports"
):
    """Generate complete monitoring report."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    detector = DriftDetector()
    detector.set_reference(X_ref, y_ref)
    
    # Data drift
    data_drift = detector.detect_data_drift(
        X_current, y_current,
        output_path=os.path.join(output_dir, "data_drift.html")
    )
    
    # Quality tests
    quality = detector.run_quality_tests(X_current)
    
    # Prediction drift (if provided)
    prediction_drift = None
    if y_pred_proba_ref is not None and y_pred_proba_current is not None:
        prediction_drift = detector.detect_prediction_drift(
            y_ref, y_current,
            y_pred_proba_ref, y_pred_proba_current,
            output_path=os.path.join(output_dir, "prediction_drift.html")
        )
    
    return {
        'data_drift': data_drift,
        'quality': quality,
        'prediction_drift': prediction_drift,
        'reports_location': output_dir
    }
"""Alert system for model degradation and drift."""
import logging
import pandas as pd
import numpy as np
from enum import Enum


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Alert:
    """Alert object."""
    
    def __init__(self, severity: AlertSeverity, message: str, metric_value: float):
        self.severity = severity
        self.message = message
        self.metric_value = metric_value


class MonitoringAlerts:
    """Generate alerts based on monitoring metrics."""
    
    # Thresholds (financial industry standards)
    PSI_THRESHOLD_WARN = 0.15  # Population Stability Index warning
    PSI_THRESHOLD_CRITICAL = 0.25  # PSI critical
    GINI_THRESHOLD = 0.35  # Minimum acceptable Gini
    KS_THRESHOLD = 0.25  # Minimum acceptable KS
    MISSING_DATA_PCT_THRESHOLD = 0.05  # 5% missing data
    PREDICTION_LATENCY_THRESHOLD_MS = 500  # Max 500ms latency
    
    def __init__(self):
        self.alerts = []
    
    def check_psi(
        self,
        feature: str,
        expected_dist: np.ndarray,
        actual_dist: np.ndarray
    ) -> Alert:
        """Check Population Stability Index for drift."""
        
        psi = self._compute_psi(expected_dist, actual_dist)
        
        if psi > self.PSI_THRESHOLD_CRITICAL:
            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                message=f"CRITICAL PSI detected for {feature}: {psi:.3f} (threshold: {self.PSI_THRESHOLD_CRITICAL}). "
                        f"Retraining recommended immediately.",
                metric_value=psi
            )
        elif psi > self.PSI_THRESHOLD_WARN:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                message=f"WARNING PSI detected for {feature}: {psi:.3f} (threshold: {self.PSI_THRESHOLD_WARN}). "
                        f"Monitor closely.",
                metric_value=psi
            )
        else:
            alert = Alert(
                severity=AlertSeverity.INFO,
                message=f"PSI normal for {feature}: {psi:.3f}",
                metric_value=psi
            )
        
        logger.log(
            logging.CRITICAL if alert.severity == AlertSeverity.CRITICAL else
            logging.WARNING if alert.severity == AlertSeverity.WARNING else
            logging.INFO,
            alert.message
        )
        
        self.alerts.append(alert)
        return alert
    
    def check_model_performance(self, gini: float, ks: float) -> list:
        """Check if model performance has degraded."""
        
        alerts = []
        
        if gini < self.GINI_THRESHOLD:
            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                message=f"Model Gini degraded: {gini:.3f} < {self.GINI_THRESHOLD}. Retraining required.",
                metric_value=gini
            )
            alerts.append(alert)
            logger.critical(alert.message)
        
        if ks < self.KS_THRESHOLD:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                message=f"Model KS degraded: {ks:.3f} < {self.KS_THRESHOLD}. Review recommended.",
                metric_value=ks
            )
            alerts.append(alert)
            logger.warning(alert.message)
        
        self.alerts.extend(alerts)
        return alerts
    
    def check_data_quality(self, X: pd.DataFrame) -> list:
        """Check for data quality issues."""
        
        alerts = []
        
        # Missing values
        missing_pct = X.isnull().sum().sum() / (len(X) * len(X.columns))
        if missing_pct > self.MISSING_DATA_PCT_THRESHOLD:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                message=f"High missing data: {missing_pct:.1%} > {self.MISSING_DATA_PCT_THRESHOLD:.1%}",
                metric_value=missing_pct
            )
            alerts.append(alert)
            logger.warning(alert.message)
        
        # Duplicate rows
        duplicate_pct = X.duplicated().sum() / len(X)
        if duplicate_pct > 0.01:
            alert = Alert(
                severity=AlertSeverity.INFO,
                message=f"Duplicate rows detected: {duplicate_pct:.1%}",
                metric_value=duplicate_pct
            )
            alerts.append(alert)
        
        self.alerts.extend(alerts)
        return alerts
    
    def check_latency(self, latency_ms: float) -> Alert:
        """Check prediction latency."""
        
        if latency_ms > self.PREDICTION_LATENCY_THRESHOLD_MS:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                message=f"High prediction latency: {latency_ms:.0f}ms > {self.PREDICTION_LATENCY_THRESHOLD_MS}ms",
                metric_value=latency_ms
            )
            logger.warning(alert.message)
        else:
            alert = Alert(
                severity=AlertSeverity.INFO,
                message=f"Latency normal: {latency_ms:.0f}ms",
                metric_value=latency_ms
            )
        
        self.alerts.append(alert)
        return alert
    
    def get_critical_alerts(self) -> list:
        """Get only critical alerts."""
        return [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]
    
    def trigger_retraining(self) -> bool:
        """Check if retraining should be triggered."""
        critical_alerts = self.get_critical_alerts()
        
        if critical_alerts:
            logger.critical(f"TRIGGERING RETRAINING: {len(critical_alerts)} critical alerts")
            return True
        
        return False
    
    @staticmethod
    def _compute_psi(expected_dist: np.ndarray, actual_dist: np.ndarray) -> float:
        """Compute Population Stability Index."""
        expected_dist = np.clip(expected_dist, 1e-6, None)
        actual_dist = np.clip(actual_dist, 1e-6, None)
        
        psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
        return psi
    
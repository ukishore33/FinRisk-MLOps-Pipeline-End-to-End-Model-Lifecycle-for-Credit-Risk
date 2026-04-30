"""Financial metrics evaluation (Gini, KS, IV)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import logging


logger = logging.getLogger(__name__)


def compute_gini(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute Gini coefficient (standard in banking)."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    gini = 2 * auc(fpr, tpr) - 1
    return gini


def compute_ks(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute KS (Kolmogorov-Smirnov) statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = np.max(np.abs(tpr - fpr))
    return ks


def compute_iv(X: pd.DataFrame, y: pd.Series, feature: str, bins: int = 10) -> float:
    """
    Compute Information Value for a feature.
    IV > 0.1 = Strong predictor
    IV > 0.05 = Moderate predictor
    IV > 0.02 = Weak predictor
    """
    df = pd.DataFrame({'X': X[feature], 'y': y})
    df['bucket'] = pd.qcut(df['X'], q=bins, duplicates='drop')
    
    grouped = df.groupby('bucket', ordered=True).agg({
        'y': ['sum', 'count']
    })
    grouped.columns = ['defaults', 'total']
    grouped['non_defaults'] = grouped['total'] - grouped['defaults']
    
    # Avoid log(0)
    grouped['pct_defaults'] = (grouped['defaults'] + 0.5) / (grouped['defaults'].sum() + 0.5)
    grouped['pct_non_defaults'] = (grouped['non_defaults'] + 0.5) / (grouped['non_defaults'].sum() + 0.5)
    
    grouped['iv_component'] = (grouped['pct_defaults'] - grouped['pct_non_defaults']) * np.log(
        grouped['pct_defaults'] / (grouped['pct_non_defaults'] + 1e-8)
    )
    
    iv = grouped['iv_component'].sum()
    return iv


def compute_psi(expected_dist: np.ndarray, actual_dist: np.ndarray) -> float:
    """
    Compute Population Stability Index for drift detection.
    PSI < 0.1 = No significant shift
    PSI 0.1-0.25 = Small shift (investigate)
    PSI > 0.25 = Significant shift (retrain recommended)
    """
    # Avoid log(0)
    expected_dist = np.clip(expected_dist, 1e-6, None)
    actual_dist = np.clip(actual_dist, 1e-6, None)
    
    psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
    return psi


def validate_model_fitness(
    test_gini: float,
    test_ks: float,
    min_gini: float = 0.35,
    min_ks: float = 0.25
) -> bool:
    """Check if model meets minimum financial thresholds."""
    if test_gini < min_gini:
        logger.warning(f"Gini {test_gini:.3f} below threshold {min_gini}")
        return False
    
    if test_ks < min_ks:
        logger.warning(f"KS {test_ks:.3f} below threshold {min_ks}")
        return False
    
    logger.info(f"✓ Model fitness validated: Gini={test_gini:.3f}, KS={test_ks:.3f}")
    return True
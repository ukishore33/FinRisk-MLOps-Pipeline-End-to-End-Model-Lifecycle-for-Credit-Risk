"""Model performance tests."""
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from training.train import LoanDefaultTrainer
from training.evaluate import compute_gini, compute_ks


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'age': np.random.randint(20, 70, n_samples),
        'income_monthly': np.random.uniform(10000, 150000, n_samples),
        'loan_amount': np.random.uniform(50000, 2000000, n_samples),
        'loan_tenure_m': np.random.choice([12, 24, 36, 48, 60, 72, 84], n_samples),
        'interest_rate': np.random.uniform(5, 30, n_samples),
        'existing_loans': np.random.randint(0, 5, n_samples),
        'credit_score': np.random.randint(300, 900, n_samples),
        'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Business', 'Freelancer'], n_samples),
        'education': np.random.choice(['Under-Graduate', 'Graduate', 'Post-Graduate', 'Professional'], n_samples),
        'residence_type': np.random.choice(['Own', 'Rented', 'Family'], n_samples),
        'delinquency_30d': np.random.randint(0, 5, n_samples),
        'utilisation_pct': np.random.uniform(0, 100, n_samples),
        'months_employed': np.random.randint(0, 300, n_samples),
        'loan_purpose': np.random.choice(['Home Improvement', 'Vehicle', 'Education', 'Medical', 'Wedding', 'Debt Consolidation', 'Personal', 'Business'], n_samples),
    })
    
    # Target: 20% default rate
    y = np.random.binomial(1, 0.2, n_samples)
    
    return X, pd.Series(y)


class TestModelPerformance:
    """Test model performance metrics."""
    
    def test_gini_computation(self):
        """Test Gini coefficient computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15])
        
        gini = compute_gini(y_true, y_pred_proba)
        
        assert 0 <= gini <= 1
        assert gini > 0.5  # Should be discriminative
    
    def test_ks_computation(self):
        """Test KS statistic computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15])
        
        ks = compute_ks(y_true, y_pred_proba)
        
        assert 0 <= ks <= 1
        assert ks > 0.3  # Reasonable separation
    
    def test_model_exceeds_minimum_gini(self, sample_data):
        """Test that trained model exceeds minimum Gini threshold."""
        X, y = sample_data
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        trainer = LoanDefaultTrainer()
        
        # Simple model
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        gini = compute_gini(y_test, y_pred_proba)
        
        assert gini >= 0.35, f"Model Gini {gini} below threshold 0.35"
    
    def test_model_prediction_probability_bounds(self, sample_data):
        """Test that predictions are valid probabilities."""
        X, y = sample_data
        
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        assert np.all(y_pred_proba >= 0)
        assert np.all(y_pred_proba <= 1)
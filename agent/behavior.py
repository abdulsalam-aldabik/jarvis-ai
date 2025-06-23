from sktime.forecasting.naive import NaiveForecaster
from sktime.datasets import load_airline
import wittgenstein as lw  # ✅ Use wittgenstein instead of ripperk
import pandas as pd

def train_time_series_forecaster(y):
    # Example: fit a naive forecaster
    forecaster = NaiveForecaster(strategy="mean")
    forecaster.fit(y)
    return forecaster

def learn_rules(X, y):
    # Example: fit RIPPER-k rule learner using wittgenstein
    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        if hasattr(X, 'columns'):
            X = pd.DataFrame(X)
        else:
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Add target column to create the format wittgenstein expects
    data = X.copy()
    data['target'] = y
    
    # Create RIPPER classifier (k=2 for RIPPERk)
    clf = lw.RIPPER(k=2, prune_size=0.33, dl_allowance=64)
    clf.fit(data, class_feat='target')
    return clf

def propose_routine(candidate, user_approval_callback):
    # candidate: dict describing the routine
    # user_approval_callback: function to call for approval
    approved = user_approval_callback(candidate)
    if approved:
        print("Routine approved:", candidate)
    else:
        print("Routine rejected:", candidate)

# Example usage:
if __name__ == "__main__":
    # Time series forecasting
    y = load_airline()
    model = train_time_series_forecaster(y)
    print("Time series forecast:", model.predict([len(y), len(y)+1, len(y)+2]))

    # For RIPPER-k, create example tabular data
    import numpy as np
    
    # Example dataset for rule learning
    np.random.seed(42)
    X_example = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 100),
        'humidity': np.random.normal(60, 15, 100),
        'pressure': np.random.normal(1013, 10, 100)
    })
    
    # Create target based on some rules (for demonstration)
    y_example = ((X_example['temperature'] > 25) & 
                 (X_example['humidity'] < 50)).astype(int)
    
    # Learn rules
    rules = learn_rules(X_example, y_example)
    print("Learned rules:")
    print(rules.ruleset_)
    
    # Example of routine proposal
    def mock_user_approval(candidate):
        print(f"Do you approve this routine? {candidate}")
        return True  # Mock approval
    
    sample_routine = {
        "name": "Morning Workout",
        "time": "07:00",
        "duration": "30 minutes",
        "activities": ["stretching", "cardio"]
    }
    
    propose_routine(sample_routine, mock_user_approval)

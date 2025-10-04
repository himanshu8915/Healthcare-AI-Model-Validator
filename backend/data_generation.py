import pandas as pd
import os
import random

def generate_synthetic_csv(model_desc: str, features_desc: str, num_samples: int) -> str:
    """
    Generates a synthetic test dataset CSV.
    
    Args:
        model_desc: Description of the model
        features_desc: Description of features / parameters
        num_samples: Number of samples to generate
        
    Returns:
        path: Path to the generated CSV
    """
    # For placeholder: create 5 numeric features
    num_features = 5
    data = {}
    
    for i in range(num_features):
        feature_name = f"feature_{i+1}"
        data[feature_name] = [round(random.random(), 3) for _ in range(num_samples)]
    
    # Random binary labels (0 or 1)
    data["label"] = [random.choice([0, 1]) for _ in range(num_samples)]
    
    # Create directory if it doesn't exist
    os.makedirs("temp_data", exist_ok=True)
    
    path = f"temp_data/synthetic_test_data.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    
    return path

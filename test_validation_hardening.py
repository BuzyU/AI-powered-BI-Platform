
import unittest
import pickle
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Mocking internal state to test ModelService standalone
from app.services.model_service import ModelService

class TestModelValidation(unittest.TestCase):
    def setUp(self):
        self.service = ModelService()
        self.model_path = Path("test_model.pkl")
        self.dict_path = Path("test_dict.pkl")
        
        # Create a valid sklearn model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model.fit(X, y)
        # sklearn sets n_features_in_ automatically
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Create an invalid "model" (dict)
        with open(self.dict_path, 'wb') as f:
            pickle.dump({'predict': 'fake'}, f)

    def tearDown(self):
        if self.model_path.exists():
            self.model_path.unlink()
        if self.dict_path.exists():
            self.dict_path.unlink()

    def test_reject_dict(self):
        print("\nTesting rejection of pickled dict...")
        with self.assertRaises(ValueError) as cm:
            self.service._load_pickle_model(self.dict_path)
        print(f"Caught expected error: {cm.exception}")
        self.assertIn("primitive type", str(cm.exception))

    def test_feature_mismatch(self):
        print("\nTesting feature count mismatch...")
        # Load the valid model manually to register it
        model, info = self.service._load_pickle_model(self.model_path)
        self.service.loaded_models['test_model'] = model
        self.service.model_metadata['test_model'] = info
        
        # Model expects 2 features. Give it 3.
        X_bad = np.array([[1, 2, 3]])
        result = self.service.predict('test_model', X_bad)
        
        print(f"Result: {result}")
        self.assertIn('error', result)
        self.assertIn('Feature mismatch', result['error'])

    def test_missing_column(self):
        print("\nTesting missing columns in DataFrame...")
        # Re-train model with feature names to test name validation
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        df = pd.DataFrame({'colA': [1, 3], 'colB': [2, 4]})
        y = [0, 1]
        model.fit(df, y)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
            
        model, info = self.service._load_pickle_model(self.model_path)
        self.service.loaded_models['test_named'] = model
        self.service.model_metadata['test_named'] = info
        
        # Predict with missing colB (replaced by colC, so count is same but name is wrong)
        X_bad = pd.DataFrame({'colA': [1], 'colC': [5]})
        result = self.service.predict('test_named', X_bad)
        
        print(f"Result: {result}")
        self.assertIn('error', result)
        self.assertIn('Missing features', result['error'])
        
    def test_valid_prediction(self):
        print("\nTesting valid prediction...")
        # Load model
        model, info = self.service._load_pickle_model(self.model_path)
        self.service.loaded_models['test_valid'] = model
        self.service.model_metadata['test_valid'] = info
        
        X_good = np.array([[1, 2]])
        result = self.service.predict('test_valid', X_good)
        
        self.assertNotIn('error', result)
        self.assertEqual(len(result['predictions']), 1)

if __name__ == '__main__':
    unittest.main()

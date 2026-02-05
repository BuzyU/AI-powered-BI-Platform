
import pickle
import pandas as pd
import numpy as np
import sys

def inspect_pickle(file_path):
    print(f"--- Inspecting {file_path} ---")
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        
        print(f"Type: {type(obj)}")
        
        if hasattr(obj, 'shape'):
            print(f"Shape: {obj.shape}")
            
        if hasattr(obj, 'columns'):
            print(f"Columns: {obj.columns}")
            
        if hasattr(obj, 'predict'):
            print("Has 'predict' method: YES")
        else:
            print("Has 'predict' method: NO")
            
        print(f"Attributes: {dir(obj)[:20]}...") # Print first 20 attributes
        
        if isinstance(obj, pd.DataFrame):
            print("\nIt IS a DataFrame.")
            print("Head:\n", obj.head())
        else:
            print("\nIt is NOT a DataFrame.")

    except Exception as e:
        print(f"Error loading pickle: {e}")

if __name__ == "__main__":
    inspect_pickle("predictions_test_data.pkl")

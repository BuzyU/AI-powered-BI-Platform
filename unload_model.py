
import requests

def unload_bad_model():
    base_url = "http://localhost:8000/api"
    headers = {"x-session-id": "default_session"}
    
    # 1. List models to find the ID for predictions_test_data.pkl
    try:
        res = requests.get(f"{base_url}/models", headers=headers)
        if res.status_code != 200:
            print(f"Failed to list models: {res.text}")
            return

        data = res.json()
        print("Models:", data)
        
        target_model = None
        # Check both uploaded and loaded models
        all_models = data.get('uploaded_models', []) + data.get('loaded_models', [])
        
        for m in all_models:
            if 'predictions_test_data' in m.get('filename', '') or 'predictions_test_data' in m.get('id', ''):
                target_model = m
                break
        
        if target_model:
            model_id = target_model.get('id')
            print(f"Found model to unload: {model_id}")
            
            # 2. Unload it
            unload_res = requests.delete(f"{base_url}/models/{model_id}", headers=headers)
            print(f"Unload response: {unload_res.status_code} {unload_res.text}")
        else:
            print("Target model predictions_test_data.pkl not found in session.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    unload_bad_model()

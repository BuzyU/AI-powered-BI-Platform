# ML Model Service - Load, Evaluate, and Predict with .pkl, .h5, .onnx models
# type: ignore[import]  # Optional ML framework imports are handled at runtime
import logging
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
from datetime import datetime

# Optional ML framework imports - these are only used if installed
# They are imported dynamically in the respective methods with proper error handling
if TYPE_CHECKING:
    import tensorflow as tf  # noqa: F401
    import torch  # noqa: F401
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401

logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and running ML models - similar to Streamlit functionality."""
    
    SUPPORTED_EXTENSIONS = {'.pkl', '.h5', '.onnx', '.pt', '.joblib'}
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
    
    def load_model(self, file_path: str, model_id: str) -> Dict[str, Any]:
        """
        Load a model from file and return metadata.
        Supports: .pkl, .h5, .onnx, .pt, .joblib
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported model format: {ext}")
        
        model = None
        model_type = "unknown"
        model_info = {}
        
        try:
            if ext in ['.pkl', '.joblib']:
                model, model_info = self._load_pickle_model(path)
                model_type = "sklearn/pickle"
                
            elif ext == '.h5':
                model, model_info = self._load_keras_model(path)
                model_type = "keras/tensorflow"
                
            elif ext == '.onnx':
                model, model_info = self._load_onnx_model(path)
                model_type = "onnx"
                
            elif ext == '.pt':
                model, model_info = self._load_pytorch_model(path)
                model_type = "pytorch"
            
            # Store model
            self.loaded_models[model_id] = model
            
            # Create metadata
            metadata = {
                'id': model_id,
                'filename': path.name,
                'model_type': model_type,
                'file_size_mb': round(path.stat().st_size / (1024 * 1024), 2),
                'loaded_at': datetime.now().isoformat(),
                'status': 'loaded',
                **model_info
            }
            
            self.model_metadata[model_id] = metadata
            logger.info(f"Loaded model {model_id}: {model_type}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {file_path}: {e}")
            return {
                'id': model_id,
                'filename': path.name,
                'status': 'error',
                'error': str(e)
            }
    
    def _load_pickle_model(self, path: Path) -> tuple:
        """Load a pickled sklearn model.
        
        WARNING: Pickle deserialization can execute arbitrary code.
        Only load models from TRUSTED sources. In production, consider:
        - Using SafeTensors for deep learning models
        - Using ONNX format for cross-framework compatibility
        - Validating file checksums before loading
        """
        logger.warning(f"Loading pickle file {path.name} - ensure this is from a trusted source")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Strict Validation against non-model types
        if isinstance(model, (dict, list, tuple, set, str, bytes, int, float, bool)):
             raise ValueError(f"Uploaded file contains a primitive type ({type(model).__name__}), not a scikit-learn model.")
        
        if isinstance(model, (np.ndarray, pd.DataFrame, pd.Series)):
             raise ValueError(f"Uploaded file contains a {type(model).__name__}, not a model. Please upload the trained model object.")

        # Validate it works like a model
        if not hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
             raise ValueError(f"Uploaded object {type(model).__name__} does not have a 'predict' method. Ensure it is a valid scikit-learn model.")

        info = {
            'framework': 'scikit-learn',
            'class': type(model).__name__,
        }
        
        # Try to extract sklearn model info
        if hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                # Limit params for display
                info['params'] = {k: str(v)[:100] for k, v in list(params.items())[:10]}
            except:
                pass
        
        if hasattr(model, 'feature_names_in_'):
            info['features'] = list(model.feature_names_in_)
        
        if hasattr(model, 'classes_'):
            info['classes'] = [str(c) for c in model.classes_]
            info['n_classes'] = len(model.classes_)
            info['task'] = 'classification'
        else:
            info['task'] = 'regression'
        
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_
            
            # Fallback: Generate generic names if missing so UI can still render inputs
            if 'features' not in info:
                info['features'] = [f"Feature {i}" for i in range(model.n_features_in_)]
            
        # Check for Pipeline
        if type(model).__name__ == 'Pipeline':
            info['is_pipeline'] = True
            info['preprocessing'] = 'internal'
        else:
            info['is_pipeline'] = False
            info['preprocessing'] = 'external'
            
        return model, info
    
    def _load_keras_model(self, path: Path) -> tuple:
        """Load a Keras/TensorFlow .h5 model."""
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(path))
            
            info = {
                'framework': 'tensorflow/keras',
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'n_layers': len(model.layers),
                'trainable_params': int(model.count_params()),
                'layers': []
            }
            
            # Extract layer info
            for i, layer in enumerate(model.layers[:20]):  # Limit to 20 layers
                layer_info = {
                    'name': layer.name,
                    'type': type(layer).__name__,
                    'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A'
                }
                info['layers'].append(layer_info)
            
            return model, info
            
        except ImportError:
            logger.warning("TensorFlow not installed, returning placeholder")
            return None, {
                'framework': 'tensorflow/keras',
                'status': 'tf_not_installed',
                'message': 'Install tensorflow to load .h5 models'
            }
    
    def _load_onnx_model(self, path: Path) -> tuple:
        """Load an ONNX model."""
        try:
            import onnx
            import onnxruntime as ort
            
            # Load with onnx for inspection
            onnx_model = onnx.load(str(path))
            
            # Create runtime session for inference
            session = ort.InferenceSession(str(path))
            
            info = {
                'framework': 'onnx',
                'ir_version': onnx_model.ir_version,
                'producer': onnx_model.producer_name,
                'inputs': [],
                'outputs': []
            }
            
            # Input info
            for inp in session.get_inputs():
                info['inputs'].append({
                    'name': inp.name,
                    'shape': inp.shape,
                    'type': inp.type
                })
            
            # Output info
            for out in session.get_outputs():
                info['outputs'].append({
                    'name': out.name,
                    'shape': out.shape,
                    'type': out.type
                })
            
            return session, info
            
        except ImportError:
            logger.warning("ONNX runtime not installed")
            return None, {
                'framework': 'onnx',
                'status': 'onnx_not_installed',
                'message': 'Install onnxruntime to load .onnx models'
            }
    
    def _load_pytorch_model(self, path: Path) -> tuple:
        """Load a PyTorch model."""
        try:
            import torch
            
            # Try loading with weights_only=True for security (prevents arbitrary code execution)
            # Set weights_only=False only for trusted legacy models
            try:
                model = torch.load(str(path), map_location='cpu', weights_only=True)
            except Exception:
                logger.warning(f"Loading {path.name} with weights_only=False - ensure trusted source")
                model = torch.load(str(path), map_location='cpu', weights_only=False)
            
            info = {
                'framework': 'pytorch',
                'type': type(model).__name__
            }
            
            if isinstance(model, dict):
                info['keys'] = list(model.keys())[:20]
                info['type'] = 'state_dict'
            elif hasattr(model, 'parameters'):
                info['n_parameters'] = sum(p.numel() for p in model.parameters())
                
            return model, info
            
        except ImportError:
            logger.warning("PyTorch not installed")
            return None, {
                'framework': 'pytorch',
                'status': 'torch_not_installed',
                'message': 'Install torch to load .pt models'
            }
    
    def predict(self, model_id: str, data: Union[pd.DataFrame, np.ndarray, List]) -> Dict[str, Any]:
        """
        Run prediction using loaded model.
        Returns predictions and probabilities if available.
        """
        if model_id not in self.loaded_models:
            return {'error': f'Model {model_id} not loaded'}
        
        model = self.loaded_models[model_id]
        metadata = self.model_metadata.get(model_id, {})
        
        # Convert input to numpy
        if isinstance(data, pd.DataFrame):
            X = data.values
            feature_names = data.columns.tolist()
        elif isinstance(data, list):
            X = np.array(data)
            feature_names = None
        else:
            X = data
            feature_names = None
        
        framework = metadata.get('framework', '')
        
        try:
            # Strict Feature Validation & Selection
            if hasattr(model, 'feature_names_in_') and feature_names:
                # 1. Try Strict Name Matching
                required_features = list(model.feature_names_in_)
                missing_features = [f for f in required_features if f not in feature_names]
                
                if not missing_features:
                    # All names match! Filter strictly (drops extra cols like target/ID)
                    if isinstance(data, pd.DataFrame):
                        X = data[required_features].values
                
                else:
                    # 2. Relaxed Fallback: Shape-based matching
                    # If names don't match, maybe the user uploaded generic 'feature1, feature2, target'
                    # Try dropping common non-feature columns and checking count
                    logger.warning(f"Feature name mismatch. Missing: {missing_features}. Attempting shape-based fallback.")
                    
                    if isinstance(data, pd.DataFrame):
                        # Drop known target/metadata columns to isolate likely features
                        # Drop known target/metadata columns to isolate likely features
                        ignore_cols = {'target', 'label', 'y', 'actual', 'class', 'id', 'index'}
                        likely_features = [
                            c for c in data.columns 
                            if c.lower() not in ignore_cols and not c.lower().startswith('unnamed')
                        ]
                        
                        # Use all numeric columns from the filtered set
                        X_candidate = data[likely_features].select_dtypes(include=[np.number]).values
                        
                        if X_candidate.shape[1] == model.n_features_in_:
                            # Success! The count matches. Use it (ignoring names).
                            X = X_candidate
                        else:
                            # Fallback failed. Raise the original name error.
                            raise ValueError(f"Missing features: {', '.join(missing_features[:5])}. (Shape valid check also failed: got {X_candidate.shape[1]} features, expected {model.n_features_in_})")
                    else:
                        # List/Array input - we can't filter by name, so we'll fall through to shape check
                        pass

            # Final Shape Check (Safety Net)
            if hasattr(model, 'n_features_in_') and X.shape[1] != model.n_features_in_:
                raise ValueError(f"Feature mismatch: Model expects {model.n_features_in_} features, but input has {X.shape[1]}.")

            if 'sklearn' in framework or 'pickle' in framework or 'scikit-learn' in framework:
                return self._predict_sklearn(model, X, metadata)
            elif 'keras' in framework or 'tensorflow' in framework:
                return self._predict_keras(model, X, metadata)
            elif 'onnx' in framework:
                return self._predict_onnx(model, X, metadata)
            elif 'pytorch' in framework:
                return self._predict_pytorch(model, X, metadata)
            else:
                return {'error': f'Unknown model framework: {framework}'}
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}
    
    def _predict_sklearn(self, model, X: np.ndarray, metadata: Dict) -> Dict:
        """Predict with sklearn model."""
        result = {
            'framework': 'sklearn',
            'n_samples': len(X)
        }
        
        # Predictions
        predictions = model.predict(X)
        result['predictions'] = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
        
        # Probabilities (for classifiers)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            result['probabilities'] = proba.tolist()
            result['confidence'] = float(np.max(proba, axis=1).mean())
            
            if 'classes' in metadata:
                result['class_labels'] = metadata['classes']
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            result['feature_importances'] = model.feature_importances_.tolist()
        
        return result
    
    def _predict_keras(self, model, X: np.ndarray, metadata: Dict) -> Dict:
        """Predict with Keras model."""
        if model is None:
            return {'error': 'TensorFlow not installed'}
        
        predictions = model.predict(X, verbose=0)
        
        result = {
            'framework': 'keras',
            'n_samples': len(X),
            'predictions': predictions.tolist()
        }
        
        # Check if classification (softmax output)
        if predictions.shape[-1] > 1:
            result['predicted_classes'] = np.argmax(predictions, axis=-1).tolist()
            result['confidence'] = float(np.max(predictions, axis=-1).mean())
        
        return result
    
    def _predict_onnx(self, session, X: np.ndarray, metadata: Dict) -> Dict:
        """Predict with ONNX model."""
        if session is None:
            return {'error': 'ONNX runtime not installed'}
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: X.astype(np.float32)})
        
        result = {
            'framework': 'onnx',
            'n_samples': len(X),
            'predictions': outputs[0].tolist() if len(outputs) > 0 else []
        }
        
        if len(outputs) > 1:
            result['additional_outputs'] = [o.tolist() for o in outputs[1:]]
        
        return result
    
    def _predict_pytorch(self, model, X: np.ndarray, metadata: Dict) -> Dict:
        """Predict with PyTorch model."""
        try:
            import torch
            
            # If model is a state_dict, can't predict
            if isinstance(model, dict):
                return {'error': 'Model is a state_dict, cannot run inference. Load the full model.'}
            
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = model(X_tensor)
                predictions = outputs.numpy()
            
            return {
                'framework': 'pytorch',
                'n_samples': len(X),
                'predictions': predictions.tolist()
            }
            
        except Exception as e:
            return {'error': f'PyTorch inference failed: {str(e)}'}
    
    def evaluate(self, model_id: str, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model accuracy on test data.
        Returns metrics like accuracy, precision, recall, F1, MSE, R2.
        """
        if model_id not in self.loaded_models:
            return {'error': f'Model {model_id} not loaded'}
        
        predictions = self.predict(model_id, X)
        if 'error' in predictions:
            return predictions
        
        y_pred = np.array(predictions['predictions'])
        metadata = self.model_metadata.get(model_id, {})
        task = metadata.get('task', 'classification')
        
        metrics = {
            'model_id': model_id,
            'n_samples': len(y_true)
        }
        
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                mean_squared_error, r2_score, mean_absolute_error,
                confusion_matrix, classification_report
            )
            
            if task == 'classification':
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
                
                # Per-class report
                try:
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    metrics['classification_report'] = report
                except:
                    pass
                    
            else:  # Regression
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
                metrics['r2'] = float(r2_score(y_true, y_pred))
                
        except ImportError:
            metrics['error'] = 'sklearn not installed for metrics calculation'
        
        return metrics
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get metadata for a loaded model."""
        return self.model_metadata.get(model_id)
    
    def list_models(self) -> List[Dict]:
        """List all loaded models."""
        return list(self.model_metadata.values())
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model to free memory."""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            del self.model_metadata[model_id]
            return True
        return False


# Singleton instance
model_service = ModelService()

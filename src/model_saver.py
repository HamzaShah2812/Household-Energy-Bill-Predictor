# src/model_saver.py

import joblib
import os

class ModelSaver:
    def __init__(self, model_dir="models", filename="svm_model.joblib"):
        self.model_dir = model_dir
        self.filename = filename

    def save(self, model):
        # Create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, self.filename)
        joblib.dump(model, path)
        print(f"✅ Model saved at: {path}")

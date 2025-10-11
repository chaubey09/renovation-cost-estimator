# src/service/io.py
import os
import joblib

def load_model(path: str):
    """
    Return a loaded model, or None if the file doesn't exist.
    Do NOT raise at import time so platforms like Render can boot the API.
    """
    if not path or not os.path.exists(path):
        # Gracefully indicate "no model yet"
        return None
    return joblib.load(path)

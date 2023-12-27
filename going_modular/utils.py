"""
Contains utility functions
"""
import torch
from pathlib import Path

def save_model(model, target_dir, model_name):
    MODEL_PATH = Path(target_dir)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = MODEL_PATH/model_name
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

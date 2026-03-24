import torch
import torchvision.models as models
import os

MODEL_PATH = r"C:\Users\darsh\OneDrive\Desktop\EYE\MODEL\vit_eye_model.pth"

def inspect_model():
    try:
        # Try loading the entire model first
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        print("Model loaded successfully!")
        print(f"Type: {type(model)}")
        
        if isinstance(model, dict):
            print("Model is a state dictionary.")
            print("Keys:", model.keys())
        else:
            print("Model is a full object.")
            # maximize recursion depth to print architecture if needed, but let's look for specific attributes
            # usually model.heads or model.classifier
            print(model)
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect_model()

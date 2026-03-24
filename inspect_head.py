import torch

MODEL_PATH = r"C:\Users\darsh\OneDrive\Desktop\EYE\MODEL\vit_eye_model.pth"

def inspect_head():
    try:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        if 'head.weight' in state_dict:
            print(f"Head Weight Shape: {state_dict['head.weight'].shape}")
        elif 'heads.head.weight' in state_dict:
            print(f"Head Weight Shape: {state_dict['heads.head.weight'].shape}")
        elif 'classifier.weight' in state_dict:
             print(f"Classifier Weight Shape: {state_dict['classifier.weight'].shape}")
        else:
            print("Could not find standard head/classifier weights. Keys:")
            # Print last few keys to guess
            print(list(state_dict.keys())[-10:])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_head()

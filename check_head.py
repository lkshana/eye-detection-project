import torch
path = r'C:\Users\darsh\OneDrive\Desktop\EYE\MODEL\vit_eye_model.pth'
try:
    sd = torch.load(path, map_location='cpu')
    # Find head weight
    head_keys = [k for k in sd.keys() if 'head.weight' in k]
    if head_keys:
        print(f"Head key found: {head_keys[0]}")
        print(f"Shape: {sd[head_keys[0]].shape}")
    else:
        print("No head key found")
        # Print all keys to be sure
        # print(list(sd.keys()))
except Exception as e:
    print(f"Error: {e}")

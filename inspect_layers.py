import torch
from torchvision.models import vit_b_16

def inspect_layers():
    try:
        model = vit_b_16(weights=None)
        print("Model structure (truncated):")
        # Print top-level modules
        for name, module in model.named_children():
            print(f"Module: {name}")
            if name == 'encoder':
                print("  Inside encoder:")
                for subname, submodule in module.named_children():
                    print(f"    Submodule: {subname}")
                    if subname == 'layers':
                         print(f"      Layer count: {len(submodule)}")
                         print(f"      First layer structure: {submodule[0]}")
                         
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_layers()

from ai_engine import analyze_image_real
import os

# Use an image from dataset if available, else create dummy
TEST_IMG = r"C:\Users\darsh\OneDrive\Desktop\EYE\MODEL\dataset\normal\1034_left.jpg"
OUTPUT_DIR = "static/uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(TEST_IMG):
    print(f"Testing with {TEST_IMG}...")
    try:
        result = analyze_image_real(TEST_IMG, OUTPUT_DIR)
        print("Result:", result)
    except Exception as e:
        print(f"Test Failed: {e}")
else:
    print("Test image not found. Please check path.")

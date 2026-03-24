import os
import random
from ai_engine import analyze_image_real

DATASET_ROOT = r"C:\Users\darsh\OneDrive\Desktop\EYE\MODEL\dataset"
OUTPUT_DIR = "static/uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Folder names in the directory (exact)
FOLDERS = [
    "Retinal Disease",
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal",
    "retinoblastoma"
]

log_file = "verification_log.txt"
with open(log_file, "w") as f:
    f.write(f"{'Folder':<25} | {'Predicted':<25} | {'Score':<10} | {'Match'}\n")
    f.write("-" * 75 + "\n")

    correct_count = 0
    total_count = 0

    for folder in FOLDERS:
        folder_path = os.path.join(DATASET_ROOT, folder)
        
        # Handle 'Retinal Disease' having a subfolder 'Training'
        if folder == "Retinal Disease":
            sub_path = os.path.join(folder_path, "Training")
            if os.path.isdir(sub_path):
                folder_path = sub_path
                
        if not os.path.isdir(folder_path):
            f.write(f"Skipping {folder} (not found)\n")
            continue
            
        files = [x for x in os.listdir(folder_path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            f.write(f"Skipping {folder} (no images)\n")
            continue
            
        # Test 5 random images
        sample_size = min(5, len(files))
        test_files = random.sample(files, sample_size)
        
        f.write(f"\n--- Testing {folder} ({sample_size} samples) ---\n")
        
        folder_correct = 0
        
        for test_file in test_files:
            file_path = os.path.join(folder_path, test_file)
            
            try:
                result = analyze_image_real(file_path, OUTPUT_DIR)
                predicted = result.get("disease_type", "Error")
                score = result.get("confidence_score", "0%")
                
                match = False
                # Improved matching logic
                # Clean strings: lowercase, remove spaces, remove underscores
                cls_clean = folder.lower().replace(" ", "").replace("_", "")
                pred_clean = predicted.lower().replace(" ", "").replace("_", "")
                
                if cls_clean == pred_clean:
                    match = True
                
                # Special cases if needed
                if folder == "Retinal Disease" and predicted == "Retinal Disease": match = True
                
                match_str = "MATCH" if match else "MISMATCH"
                f.write(f"{test_file[:15]:<20} | {predicted:<20} | {score:<8} | {match_str}\n")
                
                if match:
                    folder_correct += 1
                    correct_count += 1
                total_count += 1
                
            except Exception as e:
                f.write(f"{test_file:<20} | Error: {str(e)}\n")
        
        f.write(f"Folder Accuracy: {folder_correct}/{sample_size}\n")

    f.write("-" * 75 + "\n")
    if total_count > 0:
        f.write(f"Total Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)\n")
    else:
        f.write("Total Accuracy: 0/0\n")
    
print(f"Verification complete. Check {log_file}")

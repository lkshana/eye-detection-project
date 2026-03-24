import random
import os
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DISEASES = [
    {
        "name": "Diabetic Retinopathy",
        "risk": "High",
        "treatment": "Laser surgery, Vitrectomy, Anti-VEGF injections.",
        "suggestion": "Consult an ophthalmologist immediately. Monitor blood sugar levels.",
        "localization": "Retina (Blood Vessels)"
    },
    {
        "name": "Glaucoma",
        "risk": "High",
        "treatment": "Eye drops, Laser therapy, Surgery (Trabeculectomy).",
        "suggestion": "Regular eye pressure checks are crucial. Consult a specialist.",
        "localization": "Optic Nerve (Optic Disc)"
    },
    {
        "name": "Cataract",
        "risk": "Moderate",
        "treatment": "Cataract surgery (Lens replacement).",
        "suggestion": "Surgery is the only effective treatment. Schedule a consultation.",
        "localization": "Lens (Clouding)"
    },
    {
        "name": "Corneal Ulcer",
        "risk": "High",
        "treatment": "Antibiotic eye drops, Antifungal medication, Corneal transplant (severe cases).",
        "suggestion": "Immediate medical attention required to prevent vision loss.",
        "localization": "Cornea (Surface)"
    },
    {
        "name": "Normal",
        "risk": "Low",
        "treatment": "Routine eye exams.",
        "suggestion": "Maintain good eye hygiene and regular check-ups.",
        "localization": "N/A (Healthy Eye)"
    },
    {
        "name": "Retinitis Pigmentosa",
        "risk": "High",
        "treatment": "Vitamin A palmitate, Visual aids, Gene therapy (experimental).",
        "suggestion": "Genetic counseling and low vision aids are recommended.",
        "localization": "Retina (Photoreceptors)"
    },
    {
        "name": "Retinoblastoma",
        "risk": "Critical",
        "treatment": "Chemotherapy, Laser therapy, Cryotherapy, Surgery.",
        "suggestion": "Urgent oncological and ophthalmological evaluation needed.",
        "localization": "Retina (Tumor)"
    }
]

def generate_gaussian_heatmap(width, height, centers, sigma=100):
    """Generates a queryable 2D Gaussian heatmap array."""
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    y = y[:, np.newaxis]
    
    heatmap = np.zeros((height, width), dtype=float)
    
    for cx, cy in centers:
        # Add gaussian at (cx, cy)
        # 1/(2*pi*sigma^2) * exp( -((x-mu)^2 + (y-mu)^2) / (2*sigma^2) )
        # Simplified: exp( -dist / 2sigma^2 )
        gauss = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        heatmap += gauss
        
    # Normalize to 0-1
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
        
    return heatmap

def analyze_image(image_path, output_folder):
    """
    Simulates AI analysis of an eye image.
    Generates a realistic smooth gradient heatmap using matplotlib.
    """
    
    # 1. Randomly select a disease
    diagnosis = random.choice(DISEASES)
    
    # 2. Generate a simulated heatmap
    try:
        original = Image.open(image_path).convert("RGB") # Ensure RGB
        width, height = original.size
        
        # Decide on heatmap intensity and spots
        if diagnosis["name"] == "Normal":
            # For normal, just very faint, spread out, cool interactions
            centers = []
            heatmap_intensity = 0.0 # No intense heat
        else:
            # Random centers for disease spots
            num_centers = random.randint(1, 3)
            centers = []
            for _ in range(num_centers):
                cx = random.randint(int(width * 0.2), int(width * 0.8))
                cy = random.randint(int(height * 0.2), int(height * 0.8))
                centers.append((cx, cy))
            heatmap_intensity = random.uniform(0.6, 1.0)
            
        # Generate the activation map
        if not centers:
            # Flat cool map
            activation_map = np.zeros((height, width))
        else:
            sigma = min(width, height) / 6 # Controls spread
            activation_map = generate_gaussian_heatmap(width, height, centers, sigma=sigma)
            
        # Apply Colormap (Jet)
        colormap = plt.get_cmap('jet')
        heatmap_rgba = colormap(activation_map) # Shape: (H, W, 4)
        
        # Adjust Alpha channel based on activation intensity
        # We want low activation to be transparent, high activation to be semi-opaque
        # heat_alpha = activation_map * 0.7  (Max opacity 0.7)
        
        # Create an RGBA image
        heatmap_data = (heatmap_rgba * 255).astype(np.uint8)
        
        # Update Alpha channel: 
        # Make low values (blue-ish in Jet) transparent
        # Threshold: activation < 0.2 -> 0 alpha? 
        # Or just linear scaling of alpha.
        
        alpha_channel = (activation_map * 180).astype(np.uint8) # Max alpha ~180 (out of 255)
        # However, Jet's low values are blue. We might want to use a different colormap like 'Reds' or 'hot' if we are doing transparency?
        # But user asked for "localized". Jet with transparency works too (Rainbow effect).
        # Let's try 'jet' but with alpha scaling.
        
        heatmap_data[..., 3] = alpha_channel # Set alpha
        
        heatmap_image = Image.fromarray(heatmap_data).resize((width, height))
        
        # Composite
        original_rgba = original.convert("RGBA")
        combined = Image.alpha_composite(original_rgba, heatmap_image)
        combined = combined.convert("RGB")

        
        # ALTERNATE VISUALIZATION:
        # Keep original image, but just add the color overlay.
        # The 'jet' colormap goes Blue -> Red.
        # Areas with 0 activation are Blue. Areas with 1 are Red.
        # This matches the sample image provided which has a blue background.
        
        heatmap_filename = f"heatmap_{os.path.basename(image_path)}"
        heatmap_path = os.path.join(output_folder, heatmap_filename)
        combined.save(heatmap_path)
        
        relative_heatmap_path = f"static/uploads/{heatmap_filename}"
        
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        relative_heatmap_path = None

    # 3. Construct Result
    result = {
        "disease_type": diagnosis["name"],
        "localization": diagnosis["localization"],
        "confidence_score": f"{random.uniform(85.0, 99.9):.2f}%",
        "risk_level": diagnosis["risk"],
        "treatment_suggestion": diagnosis["treatment"],
        "first_aid": diagnosis["suggestion"],
        "heatmap_path": relative_heatmap_path
    }
    
    return result

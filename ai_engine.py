import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyC0AgR_SY3sTnZnItC3-plTSOf3XKcKpb0")
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODEL", "vit_eye_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Labels
CLASSES = [
    "Retinal Disease", "Cataract", "Diabetic Retinopathy", 
    "Glaucoma", "Normal", "Retinoblastoma"
]

DISEASE_INFO = {
    "Retinal Disease": {
        "risk": "Variable", 
        "treatment": "<ul><li><b>Comprehensive retinal exam</b> required immediately.</li><li>Treatments may include <b>laser therapy</b> to repair tears.</li><li><b>Intravitreal injections</b> (e.g., Anti-VEGF) to reduce swelling.</li><li><b>Vitrectomy surgery</b> for severe structural damage.</li></ul>", 
        "suggestion": "<ul><li><b>Avoid rubbing</b> the eyes or making sudden head movements.</li><li>Wear <b>sunglasses</b> to reduce severe light sensitivity.</li><li><b>Do not use</b> over-the-counter eye drops without consulting a professional.</li><li><b>Seek immediate care</b> if experiencing sudden vision loss, a wave of floaters, or flashes of light.</li></ul>", 
        "localization": "peripheral retinal", "affected_layers": ["Retina", "Choroid", "Optic Nerve Layer", "Retinal Pigment Epithelium", "Photoreceptor Layer", "Inner Limiting Membrane"], "severity": "Moderate to Severe"},
    "Cataract": {
        "risk": "Moderate", 
        "treatment": "<ul><li>The definitive treatment is <b>surgical removal</b> of the cloudy lens.</li><li>Replacement with an <b>artificial clear intraocular lens (IOL)</b>.</li><li>This is a highly routine and <b>successful outpatient procedure</b>.</li></ul>", 
        "suggestion": "<ul><li>Ensure <b>adequate lighting</b> when reading or working to reduce eye strain.</li><li>Wear <b>anti-glare sunglasses</b> outdoors.</li><li>Use a <b>magnifying glass</b> if reading becomes difficult.</li><li><b>Avoid night driving</b> if glare from headlights is bothersome.</li><li>Schedule a <b>surgical consultation</b> when vision loss interferes with daily activities.</li></ul>", 
        "localization": "crystalline lens", "affected_layers": ["Lens Capsule", "Lens Cortex", "Lens Nucleus", "Anterior Lens Capsule", "Posterior Lens Capsule", "Lens Epithelium"], "severity": "Progressive"},
    "Diabetic Retinopathy": {
        "risk": "High", 
        "treatment": "<ul><li>Long-term treatment requires strict <b>glycemic and blood pressure control</b>.</li><li>Focal or scatter <b>laser photocoagulation</b> to seal leaking vessels.</li><li><b>Anti-VEGF injections</b> to reduce macular swelling and prevent abnormal new vessel growth.</li></ul>", 
        "suggestion": "<ul><li>Immediately check and <b>stabilize blood sugar levels</b> according to your physician.</li><li><b>Avoid heavy lifting</b> or intense straining which can increase blood pressure.</li><li>Prevent spikes in pressure to mitigate the <b>risk of acute hemorrhage</b>.</li><li>Keep all <b>scheduled eye exams</b>, as early stages often have no symptoms.</li></ul>", 
        "localization": "peripheral retinal", "affected_layers": ["Neovascularization", "Cotton Wool Spots", "Hemorrhages", "Hard Exudates", "Microaneurysms", "Macular Region", "Peripheral Retina", "Retinal Blood Vessels"], "severity": "Advanced"},
    "Glaucoma": {
        "risk": "High", 
        "treatment": "<ul><li>Primary treatment focuses on <b>lowering intraocular pressure (IOP)</b>.</li><li>Use of daily <b>prescription eye drops</b> (e.g., prostaglandins, beta blockers).</li><li><b>Laser therapy</b> (Selective Laser Trabeculoplasty) if drops are insufficient.</li><li><b>Surgical interventions</b> (trabeculectomy) may be required to improve fluid drainage.</li></ul>", 
        "suggestion": "<ul><li><b>Do not skip</b> any prescribed glaucoma medications.</li><li><b>Avoid positions</b> that place your head below your heart (like certain yoga poses or heavy lifting).</li><li>Drink fluids in <b>moderate amounts</b> instead of large quantities at once.</li><li><b>Seek emergency care</b> if you experience severe eye pain, halos around lights, or nausea.</li></ul>", 
        "localization": "optic nerve head", "affected_layers": ["Optic Nerve Head", "Retinal Nerve Fiber Layer", "Ganglion Cell Layer", "Lamina Cribrosa", "Optic Disc", "Peripapillary Retina"], "severity": "Progressive"},
    "Normal": {
        "risk": "Low", 
        "treatment": "<ul><li><b>No medical treatment</b> is currently required.</li><li>Continue with <b>routine comprehensive eye examinations</b>.</li><li>Visit your optometrist <b>every 1 to 2 years</b> (or annually if over 60) to monitor long-term ocular health.</li></ul>", 
        "suggestion": "<ul><li>Maintain <b>good eye hygiene</b>.</li><li>Rest your eyes using the <b>20-20-20 rule</b> (every 20 minutes, look at something 20 feet away for 20 seconds).</li><li><b>Protect your eyes</b> from UV light with quality sunglasses.</li><li>Eat a <b>balanced diet</b> rich in leafy greens and omega-3 fatty acids.</li></ul>", 
        "localization": "N/A", "affected_layers": [], "severity": "None"},
    "Retinoblastoma": {
        "risk": "Critical", 
        "treatment": "<ul><li>This is a critical, life-threatening condition requiring <b>urgent pediatric oncology</b> intervention.</li><li>Treatments may include <b>systemic or intra-arterial chemotherapy</b>.</li><li><b>Focal laser therapy</b> or <b>cryotherapy</b> may be used for localized control.</li><li>In advanced cases, <b>enucleation</b> (removal of the eye) is necessary to prevent tumor spread to the brain.</li></ul>", 
        "suggestion": "<ul><li>This is a <b>medical emergency</b>. Do not delay seeking specialized tertiary care.</li><li>Keep the patient <b>calm and comfortable</b>.</li><li><b>Do not apply any physical pressure</b> to the eye.</li><li>Gather all <b>medical history and recent photographs</b> (especially those showing a 'white pupil' or leukocoria) for the specialists.</li></ul>", 
        "localization": "intraocular tumor", "affected_layers": ["Retina", "Vitreous", "Optic Nerve", "Choroid", "Sclera", "Orbital Tissue", "Subretinal Space"], "severity": "Critical"}
}

# Load Model
print(f"Loading model from {MODEL_PATH}...")
try:
    import timm
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=6)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model. {e}")
    model = None

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def reshape_transform(tensor, height=14, width=14):
    if tensor.shape[1] == 197:
        result = tensor[:, 1:, :] 
    else:
        result = tensor
    result = result.transpose(1, 2)
    side = int(result.size(2) ** 0.5)
    result = result.reshape(tensor.size(0), result.size(1), side, side)
    return result

def analyze_eye_layers(heatmap, disease_type):
    """
    Analyze heatmap intensity across different eye regions/layers
    Returns layer analysis with intensity percentages
    """
    h, w = heatmap.shape
    center_y, center_x = h // 2, w // 2
    
    # Define regions for different eye layers
    regions = {
        "Central Region": (center_x - w//6, center_y - h//6, center_x + w//6, center_y + h//6),
        "Mid-Peripheral": (center_x - w//3, center_y - h//3, center_x + w//3, center_y + h//3),
        "Outer Region": (0, 0, w, h),
        "Upper Quadrant": (0, 0, w, center_y),
        "Lower Quadrant": (0, center_y, w, h),
        "Temporal Side": (center_x, 0, w, h),
        "Nasal Side": (0, 0, center_x, h)
    }
    
    layer_analysis = {}
    total_intensity = np.sum(heatmap)
    
    if total_intensity == 0:
        return {"error": "No significant activation detected"}
    
    for region_name, (x1, y1, x2, y2) in regions.items():
        # Extract region
        region_heatmap = heatmap[y1:y2, x1:x2]
        region_intensity = np.sum(region_heatmap)
        
        # Calculate percentage
        intensity_percentage = (region_intensity / total_intensity) * 100
        
        # Determine severity based on intensity
        if intensity_percentage > 40:
            severity = "Severely Affected"
        elif intensity_percentage > 25:
            severity = "Moderately Affected"
        elif intensity_percentage > 10:
            severity = "Mildly Affected"
        else:
            severity = "Minimal Impact"
        
        layer_analysis[region_name] = {
            "intensity_percentage": round(intensity_percentage, 2),
            "severity": severity,
            "peak_activation": float(np.max(region_heatmap))
        }
    
    # Get disease-specific affected layers
    disease_layers = DISEASE_INFO.get(disease_type, {}).get("affected_layers", [])
    
    # Calculate layer impact based on heatmap distribution
    layer_impact = []
    for layer in disease_layers:
        layer_lower = layer.lower()
        
        # Simulate layer impact based on region analysis
        if "central" in layer_lower or "macular" in layer_lower or "macula" in layer_lower or "fovea" in layer_lower:
            impact = float(layer_analysis.get("Central Region", {}).get("intensity_percentage", 0.0))
        elif "peripheral" in layer_lower:
            impact = float(layer_analysis.get("Mid-Peripheral", {}).get("intensity_percentage", 0.0))
        elif "optic nerve head" in layer_lower:
            # Optic Nerve Head is specifically the central core
            impact = float(layer_analysis.get("Central Region", {}).get("intensity_percentage", 0.0)) * 0.9 + float(layer_analysis.get("Temporal Side", {}).get("intensity_percentage", 0.0)) * 0.1
        elif "optic disc" in layer_lower:
            # Optic Disc is the broader area
            impact = float(layer_analysis.get("Central Region", {}).get("intensity_percentage", 0.0)) * 0.7 + float(layer_analysis.get("Nasal Side", {}).get("intensity_percentage", 0.0)) * 0.3
        elif "lamina cribrosa" in layer_lower:
            # Deep structure, correlated with central pressure
            impact = float(layer_analysis.get("Central Region", {}).get("intensity_percentage", 0.0)) * 0.6 + float(layer_analysis.get("Mid-Peripheral", {}).get("intensity_percentage", 0.0)) * 0.2
        elif "retinal nerve fiber layer" in layer_lower:
            # RNFL spreads out across the retina
            impact = float(layer_analysis.get("Mid-Peripheral", {}).get("intensity_percentage", 0.0)) * 0.6 + float(layer_analysis.get("Upper Quadrant", {}).get("intensity_percentage", 0.0)) * 0.2 + float(layer_analysis.get("Lower Quadrant", {}).get("intensity_percentage", 0.0)) * 0.2
        elif "peripapillary" in layer_lower:
            # Area immediately surrounding the disc
            impact = float(layer_analysis.get("Central Region", {}).get("intensity_percentage", 0.0)) * 0.4 + float(layer_analysis.get("Mid-Peripheral", {}).get("intensity_percentage", 0.0)) * 0.6
        elif "optic nerve" in layer_lower:
            impact = float(layer_analysis.get("Central Region", {}).get("intensity_percentage", 0.0)) * 0.8
        elif "vessel" in layer_lower or "arter" in layer_lower or "vein" in layer_lower or "capillary" in layer_lower or "neovascularization" in layer_lower:
            impact = float(layer_analysis.get("Outer Region", {}).get("intensity_percentage", 0.0)) * 0.4 + float(layer_analysis.get("Central Region", {}).get("intensity_percentage", 0.0)) * 0.4
        elif "microaneurysm" in layer_lower:
            impact = float(layer_analysis.get("Temporal Side", {}).get("intensity_percentage", 0.0))
        elif "hemorrhage" in layer_lower:
            impact = float(layer_analysis.get("Lower Quadrant", {}).get("intensity_percentage", 0.0))
        elif "exudate" in layer_lower:
            impact = float(layer_analysis.get("Nasal Side", {}).get("intensity_percentage", 0.0))
        elif "cotton wool" in layer_lower:
            impact = float(layer_analysis.get("Upper Quadrant", {}).get("intensity_percentage", 0.0))
            base_impact = np.mean([float(layer_analysis.get(region, {}).get("intensity_percentage", 0.0)) 
                             for region in ["Central Region", "Mid-Peripheral"]])
            variation = (len(layer) % 5 - 2) * 2.5
            impact = float(base_impact) + float(variation)
            
        impact = max(0.0, min(100.0, float(impact)))
        
        body_part = "Retina (Light-sensitive tissue at the back of the eye)" # Default
        if "neovascularization" in layer_lower: body_part = "Retina / Vitreous (Can rupture and bleed into the eye's clear fluid)"
        elif "cotton wool" in layer_lower: body_part = "Retinal Nerve Fiber Layer (Blocks visual signals to the brain due to restricted blood flow)"
        elif "hemorrhage" in layer_lower: body_part = "Retinal Blood Vessels (Leaking blood damages surrounding photoreceptors)"
        elif "exudate" in layer_lower: body_part = "Macula / Retina (Lipid leaks that destroy sharp, straight-ahead vision)"
        elif "microaneurysm" in layer_lower: body_part = "Retinal Capillaries (Weakened blood vessel walls prone to rupture)"
        elif "macula" in layer_lower or "fovea" in layer_lower: body_part = "Macula (Crucial for reading, driving, and central vision)"
        elif "peripheral" in layer_lower: body_part = "Peripheral Retina (Controls your side vision and night vision)"
        elif "blood vessel" in layer_lower or "arter" in layer_lower or "vein" in layer_lower: body_part = "Retinal Vascular System (Supplies oxygen to the entire eye)"
        elif "optic nerve head" in layer_lower: body_part = "Optic Nerve (The main data cable connecting the eye to the brain)"
        elif "optic disc" in layer_lower: body_part = "Blind Spot (Where nerve bundles exit the eye to the brain)"
        elif "lamina cribrosa" in layer_lower: body_part = "Sclera Base (Collagen network supporting the optic nerve)"
        elif "retinal nerve fiber layer" in layer_lower: body_part = "Retinal Surface (Axons that carry all visual data to the nervous system)"
        elif "peripapillary" in layer_lower: body_part = "Peripapillary Tissue (Crucial support ring around the optic disc)"
        elif "ganglion cell" in layer_lower: body_part = "Inner Retina (Neurons that pre-process image data for the brain)"
        elif "lens core" in layer_lower or "lens nucleus" in layer_lower: body_part = "Crystalline Lens Core (Causes severe cloudiness and central vision loss)"
        elif "anterior capsule" in layer_lower or "anterior lens capsule" in layer_lower: body_part = "Front Lens Membrane (Affects how light enters and focuses)"
        elif "posterior capsule" in layer_lower or "posterior lens capsule" in layer_lower: body_part = "Back Lens Membrane (Crucial for sharp focal resolution)"
        elif "lens cortex" in layer_lower: body_part = "Outer Lens Layer (Causes spoke-like visual distortions)"
        elif "lens epithelium" in layer_lower: body_part = "Anterior Lens Surface (Regulates lens metabolism and clarity)"
        elif "optic nerve" in layer_lower: body_part = "Optic Nerve (Total vision loss if severed from the brain)"
        elif "vitreous" in layer_lower: body_part = "Vitreous Humor (Clear gel filling the eye; affects floaters and light transmission)"
        elif "choroid" in layer_lower: body_part = "Choroid Layer (Major vascular layer supplying blood to the outer retina)"
        elif "sclera" in layer_lower: body_part = "Sclera (The white, protective outer structural wall of the eye)"
        elif "orbital tissue" in layer_lower: body_part = "Eye Socket (Muscles and fat surrounding the eye in the skull)"
        elif "subretinal space" in layer_lower: body_part = "Subretinal Space (Can accumulate fluid, causing retinal detachment)"
        elif "retinal pigment epithelium" in layer_lower: body_part = "RPE (Nourishes visual cells; failure leads to blindness)"
        elif "photoreceptor layer" in layer_lower: body_part = "Photoreceptors (The actual Rods and Cones that sense light)"
        elif "inner limiting membrane" in layer_lower: body_part = "Inner Limiting Membrane (Structural boundary holding the retina together)"

        layer_impact.append({
            "layer_name": layer,
            "impact_percentage": round(impact, 2),
            "status": "Critical" if impact > 30 else "Significant" if impact > 15 else "Minor",
            "body_part": body_part
        })
    
    return {
        "regional_analysis": layer_analysis,
        "layer_impact": layer_impact,
        "total_affected_layers": len([l for l in layer_impact if l["impact_percentage"] > 10]),
        "critical_layers": len([l for l in layer_impact if l["status"] == "Critical"])
    }

def generate_layer_images(heatmap, original_img, layer_impact, output_folder, base_filename):
    """
    Generate individual layer images with unique, anatomically accurate heatmaps
    Returns list of layer image paths
    """
    layer_images = []
    h, w = heatmap.shape
    
    # Base smoothed heatmap to act as the organic foundation
    base_heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    for i, layer in enumerate(layer_impact):
        if layer["impact_percentage"] > 0:  # Show all identified layers
            
            layer_name = layer["layer_name"]
            impact = layer["impact_percentage"] / 100.0
            layer_lower = layer_name.lower()
            
            # Start with the actual AI heatmap
            layer_heatmap = base_heatmap.copy()
            
            # Create physiological masks
            Y, X = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            # --- ADVANCED ANATOMICAL EXTRACTION FROM ORIGINAL IMAGE ---
            original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            b, g, r = cv2.split(original_img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            g_clahe = clahe.apply(g)
            
            # 1. Apply anatomical masks to the AI heatmap
            if "retina" in layer_lower and "peripheral" not in layer_lower:
                pass # Use full base heatmap
            elif "peripheral" in layer_lower:
                # Mask out the very center smoothly
                mask = 1.0 - np.exp(-(dist_from_center**2) / (2 * (min(h, w)//4)**2))
                layer_heatmap = layer_heatmap * mask
            elif "choroid" in layer_lower:
                # Deep layer = more diffuse
                layer_heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
                
            # --- START GLAUCOMA SPECIFIC MASKS ---
            elif "optic nerve head" in layer_lower:
                # The Optic Nerve Head is the absolute bright center core of the disc
                thresh_val = np.percentile(g_clahe, 98) # Find the brightest 2% 
                _, bright_mask = cv2.threshold(g_clahe, thresh_val, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) # Tight core
                bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
                bright_mask = cv2.GaussianBlur(bright_mask, (15, 15), 0) / 255.0
                # Focus the AI heatmap exclusively on the bright core
                layer_heatmap = layer_heatmap * (bright_mask * 2.5 + 0.05)
                
            elif "optic disc" in layer_lower:
                # The Optic Disc is the broader circular area
                thresh_val = np.percentile(g_clahe, 95) # Find the brightest 5%
                _, disc_mask = cv2.threshold(g_clahe, thresh_val, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
                disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_CLOSE, kernel)
                disc_mask = cv2.dilate(disc_mask, np.ones((15,15), np.uint8))
                disc_mask = cv2.GaussianBlur(disc_mask, (15, 15), 0) / 255.0
                layer_heatmap = layer_heatmap * (disc_mask * 1.5 + 0.08)
                
            elif "lamina cribrosa" in layer_lower:
                # Deep structure properly localized inside the optic nerve head (no scattered noise)
                thresh_val = np.percentile(g_clahe, 97)
                _, core_mask = cv2.threshold(g_clahe, thresh_val, 255, cv2.THRESH_BINARY)
                core_mask = cv2.dilate(core_mask, np.ones((9,9), np.uint8))
                core_mask = cv2.GaussianBlur(core_mask, (15, 15), 0) / 255.0
                layer_heatmap = layer_heatmap * (core_mask * 1.5 + 0.05)
                
            elif "retinal nerve fiber layer" in layer_lower or "nerve fiber" in layer_lower:
                # The RNFL fans out in a distinct "bow-tie" or sweeping pattern from the disc
                thresh_val = np.percentile(g_clahe, 96)
                _, disc_mask = cv2.threshold(g_clahe, thresh_val, 255, cv2.THRESH_BINARY)
                
                # Find the center of mass of the disc to anchor the sweeping pattern
                M = cv2.moments(disc_mask)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                else:
                    cx, cy = w//2, h//2
                
                # Create horizontal sweeping radiation mask
                Y_sweep, X_sweep = np.ogrid[:h, :w]
                # Enhance horizontal spread (superior/inferior arcades)
                y_dist = np.abs(Y_sweep - cy)
                x_dist = np.abs(X_sweep - cx)
                
                # Arcuate pattern mask
                sweep_mask = np.exp(-(y_dist**2) / (2 * (h//4)**2)) * np.exp(-(x_dist) / (w//2))
                sweep_mask = sweep_mask / np.max(sweep_mask)  # Normalize
                
                # Combine with original heatmap (tightened up, removed excessive 51x51 blur)
                layer_heatmap = layer_heatmap * sweep_mask * 1.5
                
            elif "peripapillary" in layer_lower:
                # An annulus (donut) strictly hugging the outside of the optic disc
                thresh_val = np.percentile(g_clahe, 95)
                _, disc_mask = cv2.threshold(g_clahe, thresh_val, 255, cv2.THRESH_BINARY)
                
                # Inner boundary (exclude the core)
                inner_mask = cv2.dilate(disc_mask, np.ones((11,11), np.uint8))
                # Outer boundary
                outer_mask = cv2.dilate(disc_mask, np.ones((45,45), np.uint8))
                
                # Create the donut ring
                annulus_mask = cv2.subtract(outer_mask, inner_mask)
                annulus_mask = cv2.GaussianBlur(annulus_mask, (21, 21), 0) / 255.0
                layer_heatmap = layer_heatmap * (annulus_mask * 1.8 + 0.05)
                
            elif "ganglion cell" in layer_lower:
                # The GCC is highly concentrated precisely in the Macula region
                central_region = g_clahe[h//4:3*h//4, w//4:3*w//4]
                macula_thresh = np.percentile(central_region, 5) # Darkest 5%
                _, macula_mask_center = cv2.threshold(central_region, macula_thresh, 255, cv2.THRESH_BINARY_INV)
                
                # Place back into full size mask
                macula_mask = np.zeros_like(g_clahe)
                macula_mask[h//4:3*h//4, w//4:3*w//4] = macula_mask_center
                
                # Dilate to cover the full macular region (tightened)
                macula_mask = cv2.dilate(macula_mask, np.ones((31,31), np.uint8))
                macula_mask = cv2.GaussianBlur(macula_mask, (31, 31), 0) / 255.0
                layer_heatmap = layer_heatmap * (macula_mask * 2.0 + 0.05)
            # --- END GLAUCOMA SPECIFIC MASKS ---
            elif "lens" in layer_lower:
                layer_heatmap = cv2.GaussianBlur(heatmap, (41, 41), 0)
            elif "macula" in layer_lower or "fovea" in layer_lower:
                # Bias central dark area
                mask = np.exp(-(dist_from_center**2) / (2 * (min(h, w)//8)**2))
                layer_heatmap = layer_heatmap * mask
            elif "vessel" in layer_lower or "arter" in layer_lower or "vein" in layer_lower or "capillary" in layer_lower or "neovascularization" in layer_lower:
                # Extract true blood vessels from the green channel
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                morph = cv2.morphologyEx(g_clahe, cv2.MORPH_OPEN, kernel, iterations=1)
                vessels_img = cv2.subtract(morph, g_clahe)
                _, vessel_mask = cv2.threshold(vessels_img, 10, 255, cv2.THRESH_BINARY)
                vessel_mask = vessel_mask / 255.0
                # Combine true vessels with AI disease heatmap
                layer_heatmap = layer_heatmap * vessel_mask * 3.0
                
                if "neovascularization" in layer_lower:
                    # Neovascularization tends to be highly chaotic fine vessels
                    edges = cv2.Canny(g_clahe, 50, 150) / 255.0
                    layer_heatmap = np.maximum(layer_heatmap, edges * base_heatmap * 2.0)
                    
            elif "microaneurysm" in layer_lower or "hemorrhage" in layer_lower:
                # True dark lesions extraction
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
                morph_close = cv2.morphologyEx(g_clahe, cv2.MORPH_CLOSE, kernel)
                dark_lesions = cv2.subtract(morph_close, g_clahe)
                _, lesion_mask = cv2.threshold(dark_lesions, 15, 255, cv2.THRESH_BINARY)
                
                if "microaneurysm" in layer_lower:
                    # Filter for small dots
                    lesion_mask = cv2.erode(lesion_mask, np.ones((2,2), np.uint8))
                    
                lesion_mask = cv2.dilate(lesion_mask, np.ones((3,3), np.uint8)) / 255.0
                layer_heatmap = layer_heatmap * lesion_mask * 4.0
                
            elif "exudate" in layer_lower or "cotton" in layer_lower:
                # True bright lesions extraction
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
                morph_open = cv2.morphologyEx(g_clahe, cv2.MORPH_OPEN, kernel)
                bright_lesions = cv2.subtract(g_clahe, morph_open)
                
                thresh_val = 25 if "cotton" in layer_lower else 35
                _, bright_mask = cv2.threshold(bright_lesions, thresh_val, 255, cv2.THRESH_BINARY)
                bright_mask = bright_mask / 255.0
                
                if "cotton" in layer_lower:
                    bright_mask = cv2.GaussianBlur(bright_mask, (11, 11), 0)
                    
                layer_heatmap = layer_heatmap * bright_mask * 3.5
                
            elif "sclera" in layer_lower:
                # Outer bounds
                mask = dist_from_center > (min(h, w) // 2.5)
                layer_heatmap = layer_heatmap * mask
                
            # 2. Normalize organically
            if np.max(layer_heatmap) > 0:
                layer_heatmap = layer_heatmap / np.max(layer_heatmap)
                
            # Scale intensity by actual AI impact score
            layer_heatmap = layer_heatmap * min(1.0, impact * 1.5 + 0.2)
            
            # 3. Select appropriate colormap
            if "retina" in layer_lower or "macular" in layer_lower or "fovea" in layer_lower:
                colormap = cv2.COLORMAP_JET
            elif "optic nerve" in layer_lower or "disc" in layer_lower:
                colormap = cv2.COLORMAP_SPRING
            elif "vessel" in layer_lower or "arter" in layer_lower or "vein" in layer_lower or "microaneurysm" in layer_lower or "hemorrhage" in layer_lower or "neovascularization" in layer_lower:
                colormap = cv2.COLORMAP_HOT
            elif "lens" in layer_lower or "exudate" in layer_lower or "cotton" in layer_lower:
                colormap = cv2.COLORMAP_BONE
            else:
                colormap = cv2.COLORMAP_OCEAN
                
            layer_heatmap_uint8 = np.uint8(255 * layer_heatmap)
            layer_colored = cv2.applyColorMap(layer_heatmap_uint8, colormap)
            
            # 4. Organic Alpha Blending!
            # Use power curve so peaks are intensely opaque but edges are very soft
            alpha = np.clip(np.power(layer_heatmap, 0.8) * 2.5, 0, 1.0)
            alpha_bgr = np.stack([alpha]*3, axis=-1)
            
            layer_blended = (alpha_bgr * layer_colored + (1.0 - alpha_bgr) * original_img).astype(np.uint8)
            
            # Save layer image
            layer_filename = f"layer_{i+1}_{layer_name.replace(' ', '_').replace('/', '_')}_{base_filename}"
            layer_path = os.path.join(output_folder, layer_filename)
            Image.fromarray(cv2.cvtColor(layer_blended, cv2.COLOR_BGR2RGB)).save(layer_path)
            
            layer_images.append({
                "layer_number": i + 1,
                "layer_name": layer_name,
                "image_path": f"static/uploads/{layer_filename}",
                "impact_percentage": layer["impact_percentage"],
                "status": layer["status"]
            })
    
    return layer_images

def generate_gemini_report(disease, confidence, severity, layers):
    if disease == "Normal":
        return DISEASE_INFO["Normal"]["treatment"], DISEASE_INFO["Normal"]["suggestion"], "<ul><li>✔ Scan shows healthy ocular structures &rarr; No disease detected.</li></ul>"
        
    layer_names = [str(l.get('layer_name', '')) for l in layers if isinstance(l, dict) and str(l.get('status', '')) != 'Normal']
    
    prompt = f"""
You are an expert, highly advanced AI ophthalmologist analyzing a patient's deep-learning eye scan.
The Computer Vision model has detected:
- Disease: {disease}
- Confidence: {confidence}
- Severity: {severity}
- Affected Anatomical Layers: {', '.join(layer_names)}

Write a highly personalized, medical-grade report. You MUST provide exactly THREE sections.
1. Treatment Suggestions
2. First Aid / Home Advice
3. Disease Confusion Prevention Engine

Rules:
- Format ALL THREE sections as strict HTML `<ul><li>` bulleted lists.
- Each section MUST contain EXACTLY 4 bullet points. No more, no less.
- Each bullet point MUST be exactly 1 simple sentence. Do NOT write paragraphs.
- You MUST use raw HTML `<b>` to highlight the main keywords in each point. Do NOT use markdown (like **).
- Do not include headers like "<h1>Treatment</h1>", ONLY provide the `<ul>` lists.
- For Section 3 (Disease Confusion Prevention Engine), explain why it is NOT another disease based on the layer and image evidence. Use the format `<li>✔ <b>[Observation]</b> &rarr; [Disease] unlikely</li>`. Example: `<li>✔ <b>No optic nerve cupping</b> &rarr; Glaucoma unlikely</li>`.
- Separate the THREE lists with this exact string: `|||SPLIT|||`
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        parts = response.text.split("|||SPLIT|||")
        if len(parts) >= 3:
            return parts[0].strip(), parts[1].strip(), parts[2].strip()
        elif len(parts) == 2:
            return parts[0].strip(), parts[1].strip(), "<ul><li>✔ Advanced AI reasoning unavailable for this scan.</li></ul>"
        else:
            return DISEASE_INFO.get(disease, {}).get("treatment", ""), DISEASE_INFO.get(disease, {}).get("suggestion", ""), "<ul><li>✔ Advanced AI reasoning unavailable.</li></ul>"
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return DISEASE_INFO.get(disease, {}).get("treatment", ""), DISEASE_INFO.get(disease, {}).get("suggestion", ""), "<ul><li>✔ AI reasoning service temporarily unavailable.</li></ul>"

def analyze_image_real(image_path, output_folder):
    if model is None: 
        return {
            "disease_type": "Error", 
            "localization": "Model not loaded", 
            "confidence_score": "0%", 
            "heatmap_path": None,
            "layer_analysis": {"regional_analysis": {}, "layer_impact": [], "total_affected_layers": 0, "critical_layers": 0},
            "disease_severity": "Unknown",
            "total_affected_layers": 0,
            "critical_layers": 0,
            "risk_level": "Critical",
            "treatment_suggestion": "Model loading failed.",
            "first_aid": "Check system configuration.",
            "confusion_prevention": "System offline.",
            "layer_images": []
        }

    try:
        # 1. Inference
        img_pil = Image.open(image_path).convert('RGB')
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            
        predicted_class = CLASSES[predicted_idx.item()]
        
        # 2. Localization Setup (EigenCAM)
        target_layers = [model.blocks[-1]]
        cam = EigenCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        targets = [ClassifierOutputTarget(predicted_idx.item())]

        # --- SPOTLIGHT EIGENCAM LOGIC ---
        # 1. Get the Raw Organic Shape from EigenCAM (Principal Component)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Special handling for Normal case - minimize heatmap
        if predicted_class == "Normal":
            # For normal cases, create a very minimal heatmap
            grayscale_cam = np.ones_like(grayscale_cam) * 0.05  # Very low uniform intensity
            # Skip the spotlight enhancement for normal cases
            focused_cam = grayscale_cam
        else:
            # 2. Find the Peak (Focus Point)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grayscale_cam)
            
            # 3. Create a "Spotlight" Radial Mask
            # This forces the attention to focus on the main area and fade out, 
            # reducing noise while keeping the biological shape.
            h, w = grayscale_cam.shape
            Y, X = np.ogrid[:h, :w]
            center_y, center_x = max_loc[1], max_loc[0]
            
            # Standard deviation for the spotlight (control size)
            sigma = 40 # Adjust this to make the spotlight tighter/wider
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            radial_mask = np.exp(-(dist_from_center**2) / (2 * sigma**2))
            
            # 4. Multiply: Organic Shape * Spotlight
            # This keeps the details of the disease but kills background noise
            focused_cam = grayscale_cam * radial_mask
            
            # 5. Normalize
            if np.max(focused_cam) > 0:
                focused_cam = focused_cam / np.max(focused_cam)
        
        # 6. Apply Visualization
        final_heatmap = focused_cam
        
        # Create Blue-Tinted Background (Medical/Sci-Fi Look)
        img_cv = np.array(img_pil)
        img_cv = cv2.resize(img_cv, (224, 224))
        # Get grayscale for intensity
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Create a blank BGR image
        blue_tinted_bg = np.zeros_like(img_cv)
        # Map Grayscale intensity to Blue and a bit of Green (Teal/Midnight Blue)
        blue_tinted_bg[:, :, 0] = img_gray # Blue Channel (Full Intensity)
        blue_tinted_bg[:, :, 1] = (img_gray * 0.4).astype(np.uint8) # Green Channel (40%)
        blue_tinted_bg[:, :, 2] = 0 # Red Channel (0%) to keep it cool
        
        img_bg_final = blue_tinted_bg
        
        # Mask (Show only significant heatmap)
        mask = final_heatmap > 0.05
        
        # Jet Colormap (Blue->Green->Yellow->Red)
        heatmap_uint8 = np.uint8(255 * final_heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Blend
        blended = cv2.addWeighted(heatmap_colored, 0.85, img_bg_final, 0.15, 0)
        
        # Compose final image
        final_img = img_bg_final.copy()
        mask_uint8 = np.uint8(mask * 255)
        final_img[mask] = blended[mask]
        
        # Save
        visualization = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        heatmap_filename = f"network_{os.path.basename(image_path)}"
        heatmap_path = os.path.join(output_folder, heatmap_filename)
        Image.fromarray(visualization).save(heatmap_path)
        
        relative_heatmap_path = f"static/uploads/{heatmap_filename}"
        
        # 3. Layer Analysis
        layer_analysis = analyze_eye_layers(final_heatmap, predicted_class)
        
        # 4. Generate Individual Layer Images (only for disease cases)
        base_filename = os.path.basename(image_path)
        if predicted_class == "Normal":
            layer_images = []  # No layer images for normal cases
        else:
            layer_images = generate_layer_images(final_heatmap, img_bg_final, layer_analysis["layer_impact"], output_folder, base_filename)
        
        # Get disease severity
        disease_severity = DISEASE_INFO.get(predicted_class, {}).get("severity", "Unknown")
        
        # Format string confidence
        str_conf = f"{confidence.item()*100:.2f}%"

        # Dynamically Generate Gemini AI Report
        ai_treatment, ai_firstaid, ai_confusion = generate_gemini_report(
            disease=predicted_class, 
            confidence=str_conf, 
            severity=disease_severity, 
            layers=layer_analysis.get("layer_impact", [])
        )

        raw_loc = str(DISEASE_INFO.get(predicted_class, {}).get("localization", "Unknown"))
        formatted_loc = "N/A" if raw_loc in ["N/A", "Unknown"] else f"Lesions observed in the {raw_loc.lower()} region."

        return {
            "disease_type": predicted_class,
            "localization": formatted_loc,
            "confidence_score": str_conf,
            "risk_level": DISEASE_INFO.get(predicted_class, {}).get("risk", "Unknown"),
            "treatment_suggestion": ai_treatment,
            "first_aid": ai_firstaid,
            "confusion_prevention": ai_confusion,
            "heatmap_path": relative_heatmap_path,
            "layer_analysis": layer_analysis,
            "disease_severity": disease_severity,
            "total_affected_layers": layer_analysis.get("total_affected_layers", 0),
            "critical_layers": layer_analysis.get("critical_layers", 0),
            "layer_images": layer_images
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "disease_type": "Error", 
            "localization": "Error", 
            "confidence_score": "0%", 
            "heatmap_path": None,
            "layer_analysis": {"regional_analysis": {}, "layer_impact": [], "total_affected_layers": 0, "critical_layers": 0},
            "disease_severity": "Unknown",
            "total_affected_layers": 0,
            "critical_layers": 0,
            "risk_level": "Unknown",
            "treatment_suggestion": "System error occurred.",
            "first_aid": "Please try again.",
            "confusion_prevention": "System offline.",
            "layer_images": []
        }

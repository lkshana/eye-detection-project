# 👁️ Eye Detection Project

## 📌 Overview

This project is an AI-based Eye Detection and Monitoring System that analyzes eye and head movements in real-time using computer vision techniques. It is designed to track user attention, detect movement patterns, and generate insights.

---

## 🚀 Features

* 👁️ Eye movement detection
* 🧠 Head pose tracking
* 📷 Real-time camera processing
* 🔍 AI-based analysis engine
* 📄 Report generation (PDF support)
* 📊 Visualization (heatmaps, tracking)

---

## 🛠️ Tech Stack

* Python 🐍
* OpenCV
* NumPy
* Flask (for web interface)
* Machine Learning / AI models

---

## 📁 Project Structure

```
eye-detection-project/
│
├── static/              # CSS, JS, images
├── templates/           # HTML files
├── MODEL/               # Trained model (optional / external)
│
├── app.py               # Main application
├── ai_engine.py         # Core AI logic
├── check_head.py        # Head detection
├── check_keys.py        # Input handling
├── inspect_*            # Model inspection utilities
├── pdf_generator.py     # Report generation
│
├── requirements.txt     # Dependencies
├── README.md            # Documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/eye-detection-project.git
cd eye-detection-project
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 📦 Model Download (Important)

Due to GitHub file size limits, the model is not included.

👉 Download the model from here:
**[Add your Google Drive link here]**

After downloading, place it inside:

```
MODEL/
```

---

## ⚠️ Notes

* Ensure your webcam is enabled for real-time detection
* Large files are excluded using `.gitignore`
* Works best in a well-lit environment

---

## 📌 Future Improvements

* Improve detection accuracy
* Add deep learning models
* Deploy as a web application
* Add user analytics dashboard

---

## 🙌 Acknowledgements

This project is built using open-source libraries and computer vision techniques.

---



import os
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
# from mock_ai import analyze_image # Deprecated
from ai_engine import analyze_image_real as analyze_image
from pdf_generator import create_pdf

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
REPORTS_FOLDER = 'static/reports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run Mock AI Analysis
        # We pass absolute path for processing, but return relative path for frontend
        abs_filepath = os.path.abspath(filepath)
        abs_upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
        
        analysis_result = analyze_image(abs_filepath, abs_upload_folder)
        
        # Add original image path to result for frontend display
        analysis_result['original_image'] = f"static/uploads/{filename}"
        
        # Generate PDF Report
        report_filename = f"report_{filename.rsplit('.', 1)[0]}.pdf"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        abs_report_path = os.path.abspath(report_path)
        
        # Add abs path of heatmap for PDF generator
        heatmap_filename = os.path.basename(analysis_result['heatmap_path'])
        analysis_result['heatmap_path_abs'] = os.path.join(abs_upload_folder, heatmap_filename)

        create_pdf(analysis_result, abs_report_path, abs_filepath)
        
        analysis_result['pdf_report'] = f"static/reports/{report_filename}"
        
        return render_template('result.html', result=analysis_result)

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_report(filename):
    return send_file(os.path.join(app.config['REPORTS_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

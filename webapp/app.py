import os
import sys
import threading
import uuid
import subprocess
from flask import (
    Flask, render_template, Response, jsonify, request, 
    redirect, url_for, session, flash
)
from werkzeug.utils import secure_filename

# --- App Initialization and Config ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# Secret key for session management (used for admin login and flash messages)
app.secret_key = os.urandom(24) 

# --- Admin Config ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# --- File Upload Config ---
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Note: static folder is created by Flask, but we need its path

# This will be replaced by main.py with the actual frame generator
frame_generator = None

def allowed_file(filename):
    """Checks if a file's extension is in the allowed list."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Main App Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def live():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    # This route renders the page, but the feed is at /video_feed
    # and data is at /status.
    # We'll use 'admin.html' as the main dashboard.
    return redirect(url_for('admin_dashboard'))

@app.route('/status')
def status():
    """
    Provides a JSON endpoint for the dashboard to fetch real-time data.
    """
    # Return only lightweight, JSON-serializable status fields to the UI.
    from main import shared_state
    try:
        last_count = int(shared_state.get('last_count', 0))
    except Exception:
        last_count = 0
    try:
        # Updated to send 'alert_threshold' from the new main.py
        alert_threshold = int(shared_state.get('alert_threshold', -1))
    except Exception:
        alert_threshold = -1
    
    # Send both count and the current threshold
    return jsonify({
        'last_count': last_count, 
        'alert_threshold': alert_threshold
    })

@app.route('/video_feed')
def video_feed():
    """Provides the live MJPEG stream for the admin dashboard."""
    global frame_generator
    if frame_generator is None:
        return 'Stream not started', 503
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Admin Login/Logout Routes ---

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            error = "Invalid username or password"
    return render_template('admin_login.html', error=error)

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    # admin.html should contain the video feed, status display, etc.
    return render_template('admin.html') 

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

# --- New File Processing Routes ---

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        # Create a unique filename to prevent overwrites
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{ext}"
        
        # Save the uploaded file
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # Define the output path in the static folder
        output_filename = f"result_{unique_filename}"
        output_path = os.path.join(STATIC_FOLDER, output_filename)
        
        # Get the path to the root 'process_file.py' script
        processor_script_path = os.path.join(os.path.dirname(__file__), '..', 'process_file.py')

        print(f"Calling subprocess: {sys.executable} {processor_script_path} {input_path} {output_path}")
        
        try:
            # Run the processing script
            # Use sys.executable to ensure it uses the correct python interpreter
            subprocess.run([sys.executable, processor_script_path, input_path, output_path], 
                           check=True, timeout=300) # 5 min timeout
                           
            # Redirect to the result page
            return redirect(url_for('show_result', filename=output_filename))
            
        except subprocess.CalledProcessError as e:
            print(f"Error during processing: {e}")
            flash('An error occurred while processing the file.')
            return redirect(url_for('index'))
        except subprocess.TimeoutExpired:
            print("Processing timed out.")
            flash('Processing took too long and was cancelled.')
            return redirect(url_for('index'))
            
    else:
        flash('File type not allowed')
        return redirect(url_for('index'))

@app.route('/results/<filename>')
def show_result(filename):
    """Shows the page with the processed image or video."""
    return render_template('results.html', filename=filename)

# --- Run Application ---

def run_app(port=5000):
    app.run(host='0.0.0.0', port=port, threaded=True)

if __name__ == '__main__':
    # This allows running the app directly for testing
    # but in production, it will be imported and run by main.py
    run_app(5000)

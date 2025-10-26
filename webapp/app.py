from flask import Flask, render_template, Response, jsonify
import threading, os
from flask import request, redirect, url_for, session

app = Flask(__name__, template_folder='templates', static_folder='static')

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# This will be replaced by main.py with the actual frame generator
frame_generator = None
app.secret_key = os.urandom(24)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/live')
def live():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('admin.html')

@app.route('/status')
def status():
    # Return only lightweight, JSON-serializable status fields to the UI.
    from main import shared_state
    try:
        last_count = int(shared_state.get('last_count', 0))
    except Exception:
        last_count = 0
    try:
        last_density = float(shared_state.get('last_density', 0.0))
    except Exception:
        last_density = 0.0
    return jsonify({'last_count': last_count, 'last_density': last_density})

# @app.route('/admin', methods=['GET', 'POST'])
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
    return render_template('admin.html')
  # Dashboard template will include live feed & graph
@app.route('/video_feed')
def video_feed():
    global frame_generator
    if frame_generator is None:
        return 'Stream not started', 503
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

def run_app(port=5000):
    app.run(host='0.0.0.0', port=port, threaded=True)

if __name__ == '__main__':
    run_app(5000)
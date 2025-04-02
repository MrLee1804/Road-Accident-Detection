from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import cv2
import numpy as np
import time
from datetime import datetime
import threading
import base64
from PIL import Image
import io
import json
import os
from collections import deque
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('accident_detection.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
CONFIG = {
    'motion_threshold': 25,
    'min_contour_area': 1000,
    'collision_threshold': 50,
    'object_lifetime': 30,
    'speed_calibration': 0.1,
    'alert_cooldown': 5,  # seconds
    'max_history_size': 100,
    'recording_duration': 30  # seconds
}

class Camera:
    def __init__(self, camera_id):
        # Initialize recording variables first to avoid __del__ errors
        self.is_recording = False
        self.recording_start_time = 0
        self.video_writer = None
        
        self.camera_id = camera_id
        # Try different camera backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]  # Try DirectShow first, then default
        self.cap = None
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(camera_id, backend)
                if self.cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = self.cap.read()
                    if ret:
                        logging.info(f"Successfully initialized camera {camera_id} with backend {backend}")
                        break
                    else:
                        self.cap.release()
                        logging.warning(f"Camera {camera_id} opened with backend {backend} but cannot read frames")
            except Exception as e:
                logging.error(f"Error initializing camera {camera_id} with backend {backend}: {str(e)}")
                if self.cap:
                    self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            logging.error(f"Failed to initialize camera {camera_id} with any backend")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize variables
        self.speed = 0.0
        self.acceleration = 0.0
        self.prev_frame = None
        self.prev_gray_frame = None
        self.prev_time = time.time()
        self.prev_speed = 0.0
        self.accident_detected = False
        self.last_alert_time = 0
        
        # FPS monitoring
        self.fps = 999  # Set fixed FPS to 999
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Object tracking variables
        self.prev_contours = []
        self.object_tracks = []
        self.collision_threshold = CONFIG['collision_threshold']
        self.object_lifetime = CONFIG['object_lifetime']
        
        # History tracking
        self.speed_history = deque(maxlen=CONFIG['max_history_size'])
        self.acceleration_history = deque(maxlen=CONFIG['max_history_size'])
        self.accident_history = deque(maxlen=CONFIG['max_history_size'])
        self.fps_history = deque(maxlen=CONFIG['max_history_size'])
        
        logging.info(f"Camera {camera_id} initialized successfully")
    
    def update_fps(self):
        self.frame_count += 1
        if time.time() - self.fps_start_time > 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = time.time()
            self.fps_history.append(self.fps)
    
    def get_analytics_data(self):
        try:
            speed_stats = {
                'current': float(self.speed),
                'average': float(np.mean(list(self.speed_history))) if self.speed_history else 0.0,
                'max': float(max(self.speed_history)) if self.speed_history else 0.0,
                'min': float(min(self.speed_history)) if self.speed_history else 0.0
            }
            acceleration_stats = {
                'current': float(self.acceleration),
                'average': float(np.mean(list(self.acceleration_history))) if self.acceleration_history else 0.0,
                'max': float(max(self.acceleration_history)) if self.acceleration_history else 0.0,
                'min': float(min(self.acceleration_history)) if self.acceleration_history else 0.0
            }
            fps_stats = {
                'current': int(self.fps),
                'average': float(np.mean(list(self.fps_history))) if self.fps_history else 0.0,
                'max': int(max(self.fps_history)) if self.fps_history else 0,
                'min': int(min(self.fps_history)) if self.fps_history else 0
            }
            accident_stats = {
                'total': len(self.accident_history),
                'last_24h': len([acc for acc in self.accident_history 
                                if (datetime.now() - datetime.fromisoformat(acc['timestamp'])).days < 1])
            }
            
            return {
                'speed_stats': speed_stats,
                'acceleration_stats': acceleration_stats,
                'fps_stats': fps_stats,
                'accident_stats': accident_stats
            }
        except Exception as e:
            logging.error(f"Error calculating analytics data: {str(e)}")
            return {
                'speed_stats': {'current': 0.0, 'average': 0.0, 'max': 0.0, 'min': 0.0},
                'acceleration_stats': {'current': 0.0, 'average': 0.0, 'max': 0.0, 'min': 0.0},
                'fps_stats': {'current': 0, 'average': 0.0, 'max': 0, 'min': 0},
                'accident_stats': {'total': 0, 'last_24h': 0}
            }
    
    def calculate_speed_and_acceleration(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(gray, self.prev_gray_frame)
            motion = float(np.mean(diff))  # Convert to float to avoid numpy type issues
            
            current_time = time.time()
            time_diff = current_time - self.prev_time
            
            # Calculate speed (motion * calibration factor)
            self.speed = motion * CONFIG['speed_calibration']
            self.speed_history.append(float(self.speed))  # Convert to float for storage
            
            # Calculate acceleration
            if time_diff > 0:
                self.acceleration = float((self.speed - self.prev_speed) / time_diff)
            else:
                self.acceleration = 0.0
            
            self.acceleration_history.append(float(self.acceleration))  # Convert to float for storage
            self.prev_speed = float(self.speed)  # Store as float
            self.prev_time = current_time
        
        self.prev_gray_frame = gray.copy()
        self.prev_frame = frame.copy()
    
    def detect_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray_frame is not None:
            diff = cv2.absdiff(blurred, self.prev_gray_frame)
            thresh = cv2.threshold(diff, CONFIG['motion_threshold'], 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > CONFIG['min_contour_area']]
            
            return valid_contours
        
        return []
    
    def calculate_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None
    
    def update_object_tracks(self, contours):
        current_centroids = []
        
        for contour in contours:
            centroid = self.calculate_centroid(contour)
            if centroid:
                current_centroids.append(centroid)
        
        for track in self.object_tracks:
            track['lifetime'] -= 1
            if track['lifetime'] <= 0:
                self.object_tracks.remove(track)
        
        for centroid in current_centroids:
            matched = False
            for track in self.object_tracks:
                distance = np.sqrt((centroid[0] - track['centroid'][0])**2 + 
                                 (centroid[1] - track['centroid'][1])**2)
                if distance < self.collision_threshold:
                    track['centroid'] = centroid
                    track['lifetime'] = self.object_lifetime
                    matched = True
                    break
            
            if not matched:
                self.object_tracks.append({
                    'centroid': centroid,
                    'lifetime': self.object_lifetime
                })
    
    def detect_collision(self):
        for i in range(len(self.object_tracks)):
            for j in range(i + 1, len(self.object_tracks)):
                track1 = self.object_tracks[i]
                track2 = self.object_tracks[j]
                
                distance = np.sqrt((track1['centroid'][0] - track2['centroid'][0])**2 + 
                                 (track1['centroid'][1] - track2['centroid'][1])**2)
                
                if distance < self.collision_threshold:
                    return True
        
        return False
    
    def start_recording(self, frame):
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accident_{self.camera_id}_{timestamp}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            self.is_recording = True
            self.recording_start_time = time.time()
            
            logging.info(f"Started recording accident for camera {self.camera_id}")
    
    def stop_recording(self):
        if self.is_recording:
            self.video_writer.release()
            self.is_recording = False
            logging.info(f"Stopped recording accident for camera {self.camera_id}")
    
    def detect_accident(self, frame):
        contours = self.detect_objects(frame)
        self.update_object_tracks(contours)
        collision_detected = self.detect_collision()
        
        # Draw object tracks and collision detection on frame
        for track in self.object_tracks:
            cv2.circle(frame, track['centroid'], 5, (0, 255, 0), -1)
        
        if collision_detected:
            current_time = time.time()
            if current_time - self.last_alert_time >= CONFIG['alert_cooldown']:
                self.last_alert_time = current_time
                self.accident_detected = True
                self.accident_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'speed': self.speed,
                    'acceleration': self.acceleration
                })
                self.start_recording(frame)
                logging.warning(f"Accident detected on camera {self.camera_id}")
            else:
                self.accident_detected = False
        else:
            self.accident_detected = False
            self.stop_recording()
        
        if self.is_recording:
            if time.time() - self.recording_start_time >= CONFIG['recording_duration']:
                self.stop_recording()
            else:
                self.video_writer.write(frame)
        
        return self.accident_detected
    
    def create_original_view(self, frame):
        view = frame.copy()
        cv2.putText(view, "Original Feed", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return view
    
    def create_motion_view(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray_frame is not None:
            diff = cv2.absdiff(blurred, self.prev_gray_frame)
            thresh = cv2.threshold(diff, CONFIG['motion_threshold'], 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            motion_view = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            cv2.putText(motion_view, "Motion Detection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return motion_view
        
        return frame
    
    def create_speed_view(self, frame):
        speed_view = frame.copy()
        center = (320, 240)
        radius = 100
        speed_angle = min(360, (self.speed / 100) * 360)
        
        # Draw speedometer background
        cv2.circle(speed_view, center, radius, (255, 255, 255), 2)
        cv2.circle(speed_view, center, radius-10, (0, 0, 0), -1)
        
        # Draw speed indicator
        end_x = int(center[0] + radius * np.cos(np.radians(speed_angle - 90)))
        end_y = int(center[1] + radius * np.sin(np.radians(speed_angle - 90)))
        cv2.line(speed_view, center, (end_x, end_y), (0, 255, 0), 3)
        
        # Add speed text
        cv2.putText(speed_view, f"Speed: {self.speed:.1f} km/h", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(speed_view, "Speed View", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return speed_view
    
    def create_acceleration_view(self, frame):
        accel_view = frame.copy()
        bar_width = 400
        bar_height = 40
        bar_x = 120
        bar_y = 200
        
        # Draw background bar
        cv2.rectangle(accel_view, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (255, 255, 255), 2)
        
        # Calculate acceleration bar width
        accel_width = int(min(bar_width, max(0, (self.acceleration + 10) * 20)))
        accel_color = (0, 255, 0) if self.acceleration >= 0 else (0, 0, 255)
        
        # Draw acceleration bar
        cv2.rectangle(accel_view, (bar_x, bar_y), (bar_x + accel_width, bar_y + bar_height),
                     accel_color, -1)
        
        # Add acceleration text
        cv2.putText(accel_view, f"Acceleration: {self.acceleration:.1f} m/s²", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(accel_view, "Acceleration View", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return accel_view
    
    def process_frame(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                
                self.update_fps()  # Update FPS counter
                self.calculate_speed_and_acceleration(frame)
                self.accident_detected = self.detect_accident(frame)
                
                views = {
                    'original': self.create_original_view(frame),
                    'motion': self.create_motion_view(frame),
                    'speed': self.create_speed_view(frame),
                    'acceleration': self.create_acceleration_view(frame)
                }
                
                view_data = {}
                for view_name, view_frame in views.items():
                    _, buffer = cv2.imencode('.jpg', view_frame)
                    view_data[view_name] = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    'views': view_data,
                    'speed': round(float(self.speed), 2),
                    'acceleration': round(float(self.acceleration), 2),
                    'accident_detected': self.accident_detected,
                    'camera_id': self.camera_id,
                    'timestamp': datetime.now().isoformat(),
                    'fps': int(self.fps)
                }
            return None
        except Exception as e:
            logging.error(f"Error processing frame from camera {self.camera_id}: {str(e)}")
            return None
    
    def __del__(self):
        try:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'is_recording') and self.is_recording:
                self.stop_recording()
        except Exception as e:
            logging.error(f"Error during camera {self.camera_id} cleanup: {str(e)}")

class AccidentDetectionSystem:
    def __init__(self):
        self.cameras = []
        self.total_accidents = 0
        self.last_24h_accidents = 0
        self.accident_timestamps = []
        
        # Only initialize camera 0
        camera_indices = [0]
        
        for idx in camera_indices:
            try:
                camera = Camera(idx)
                if camera.cap and camera.cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = camera.cap.read()
                    if ret:
                        logging.info(f"Successfully initialized camera {idx}")
                        self.cameras.append(camera)
                    else:
                        logging.error(f"Camera {idx} opened but cannot read frames")
                        camera.cap.release()
                else:
                    logging.error(f"Failed to initialize camera {idx}")
            except Exception as e:
                logging.error(f"Error initializing camera {idx}: {str(e)}")
        
        if not self.cameras:
            logging.warning("No cameras were successfully initialized!")
        else:
            logging.info(f"Successfully initialized {len(self.cameras)} cameras")
        
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()

    def update_accident_counts(self, accident_detected, timestamp):
        if accident_detected:
            self.total_accidents += 1
            self.accident_timestamps.append(timestamp)
            
            # Remove accidents older than 24 hours
            current_time = datetime.now()
            self.accident_timestamps = [ts for ts in self.accident_timestamps 
                                     if (current_time - ts).total_seconds() <= 24 * 3600]
            self.last_24h_accidents = len(self.accident_timestamps)

    def get_system_stats(self):
        return {
            'total_cameras': len(self.cameras),
            'active_cameras': len([cam for cam in self.cameras if cam.cap and cam.cap.isOpened()]),
            'total_accidents': self.total_accidents,
            'last_24h_accidents': self.last_24h_accidents
        }

    def detection_loop(self):
        while True:
            for camera in self.cameras:
                try:
                    if not camera.cap or not camera.cap.isOpened():
                        logging.error(f"Camera {camera.camera_id} is not opened, attempting to reinitialize")
                        camera.__init__(camera.camera_id)
                        continue
                        
                    data = camera.process_frame()
                    if data:
                        # Update accident counts
                        self.update_accident_counts(data['accident_detected'], datetime.fromisoformat(data['timestamp']))
                        
                        # Add system stats to the data
                        data['system_stats'] = self.get_system_stats()
                        
                        socketio.emit('update_data', data)
                        
                        if data['accident_detected']:
                            alert_msg = f"ALERT: Accident detected on Camera {data['camera_id']}! Speed: {data['speed']:.2f} km/h, Acceleration: {data['acceleration']:.2f} m/s²"
                            logging.warning(alert_msg)
                            
                            with open("accident_log.txt", "a") as f:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                f.write(f"{timestamp} - {alert_msg}\n")
                except Exception as e:
                    logging.error(f"Error processing frame from camera {camera.camera_id}: {str(e)}")
            
            time.sleep(0.1)  # Reduced sleep time for more responsive updates

# Initialize the accident detection system
accident_system = AccidentDetectionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    status = {
        'cameras': len(accident_system.cameras),
        'accidents_detected': sum(1 for camera in accident_system.cameras if camera.accident_detected),
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/api/camera/<int:camera_id>/history')
def get_camera_history(camera_id):
    for camera in accident_system.cameras:
        if camera.camera_id == camera_id:
            history = {
                'speed_history': list(camera.speed_history),
                'acceleration_history': list(camera.acceleration_history),
                'accident_history': list(camera.accident_history)
            }
            return jsonify(history)
    return jsonify({'error': 'Camera not found'}), 404

@app.route('/api/analytics')
def get_analytics():
    analytics_data = {
        'system': {
            'total_cameras': len(accident_system.cameras),
            'active_cameras': sum(1 for camera in accident_system.cameras if camera.cap.isOpened()),
            'total_accidents': sum(len(camera.accident_history) for camera in accident_system.cameras),
            'last_24h_accidents': sum(len([acc for acc in camera.accident_history 
                                         if (datetime.now() - datetime.fromisoformat(acc['timestamp'])).days < 1])
                                    for camera in accident_system.cameras)
        },
        'cameras': {}
    }
    
    for camera in accident_system.cameras:
        analytics_data['cameras'][camera.camera_id] = camera.get_analytics_data()
    
    return jsonify(analytics_data)

@app.route('/api/history')
def get_history():
    history_data = {
        'accidents': [],
        'system_status': []
    }
    
    # Collect accident history from all cameras
    for camera in accident_system.cameras:
        for accident in camera.accident_history:
            accident_data = accident.copy()
            accident_data['camera_id'] = camera.camera_id
            history_data['accidents'].append(accident_data)
    
    # Sort accidents by timestamp
    history_data['accidents'].sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify(history_data)

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        try:
            new_settings = request.json
            for key, value in new_settings.items():
                if key in CONFIG:
                    CONFIG[key] = value
            return jsonify({'status': 'success', 'message': 'Settings updated successfully'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    return jsonify(CONFIG)

if __name__ == '__main__':
    socketio.run(app, debug=True) 
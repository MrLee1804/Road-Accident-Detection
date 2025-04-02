import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyttsx3
import time
from datetime import datetime

class AccidentDetectionSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Road Accident Detection System")
        self.root.geometry("1200x800")
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Initialize variables
        self.speed = 0
        self.acceleration = 0
        self.prev_frame = None
        self.prev_gray_frame = None
        self.prev_time = time.time()
        self.prev_speed = 0
        self.accident_detected = False
        
        # Create GUI elements
        self.create_gui()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            tk.messagebox.showerror("Error", "Could not open camera. Please check your camera connection.")
            self.root.destroy()
            return
        
        # Start the main loop
        self.update_frame()
        self.root.mainloop()
    
    def create_gui(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Create video frame
        self.video_frame = ttk.Label(self.main_frame)
        self.video_frame.pack(pady=10)
        
        # Create info frame
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.pack(pady=10)
        
        # Speed display
        self.speed_label = ttk.Label(self.info_frame, text="Speed: 0 km/h", font=('Arial', 12))
        self.speed_label.pack(side=tk.LEFT, padx=10)
        
        # Acceleration display
        self.accel_label = ttk.Label(self.info_frame, text="Acceleration: 0 m/s²", font=('Arial', 12))
        self.accel_label.pack(side=tk.LEFT, padx=10)
        
        # Status display
        self.status_label = ttk.Label(self.info_frame, text="Status: Normal", font=('Arial', 12))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Alert button
        self.alert_button = ttk.Button(self.main_frame, text="Send Alert", command=self.send_alert)
        self.alert_button.pack(pady=10)
    
    def calculate_speed_and_acceleration(self, frame):
        # Convert frame to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(gray, self.prev_gray_frame)
            
            # Calculate motion
            motion = np.mean(diff)
            
            # Calculate time difference
            current_time = time.time()
            time_diff = current_time - self.prev_time
            
            # Calculate speed (simplified)
            self.speed = motion * 0.1  # Convert motion to speed
            
            # Calculate acceleration
            if time_diff > 0:
                self.acceleration = (self.speed - self.prev_speed) / time_diff
            else:
                self.acceleration = 0
            
            self.prev_speed = self.speed
            self.prev_time = current_time
        
        self.prev_gray_frame = gray.copy()
        self.prev_frame = frame.copy()
    
    def detect_accident(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(blurred, self.prev_gray_frame)
            
            # Threshold the difference
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate the thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for significant motion
            for contour in contours:
                if cv2.contourArea(contour) > 5000:  # Threshold for significant motion
                    return True
        
        return False
    
    def send_alert(self):
        if self.accident_detected:
            # Generate alert message
            alert_msg = f"ALERT: Accident detected! Speed: {self.speed:.2f} km/h, Acceleration: {self.acceleration:.2f} m/s²"
            
            # Update status label
            self.status_label.config(text=f"Status: {alert_msg}")
            
            # Speak alert
            self.engine.say(alert_msg)
            self.engine.runAndWait()
            
            # Save alert to file
            with open("accident_log.txt", "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} - {alert_msg}\n")
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Calculate speed and acceleration
            self.calculate_speed_and_acceleration(frame)
            
            # Detect accident
            self.accident_detected = self.detect_accident(frame)
            
            # Update GUI
            self.speed_label.config(text=f"Speed: {self.speed:.2f} km/h")
            self.accel_label.config(text=f"Acceleration: {self.acceleration:.2f} m/s²")
            
            if self.accident_detected:
                self.status_label.config(text="Status: ACCIDENT DETECTED!")
                self.status_label.config(foreground="red")
            else:
                self.status_label.config(text="Status: Normal")
                self.status_label.config(foreground="black")
            
            # Convert frame for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_frame.config(image=photo)
            self.video_frame.image = photo
        else:
            tk.messagebox.showerror("Error", "Failed to read frame from camera")
            self.root.destroy()
            return
        
        # Schedule next update
        self.root.after(10, self.update_frame)
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    AccidentDetectionSystem() 
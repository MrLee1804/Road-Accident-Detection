# Road Accident Detection System

This is a Python-based road accident detection system that uses computer vision to monitor road conditions, detect accidents, and provide real-time alerts. The system includes features for speed monitoring, acceleration tracking, and visual alerts.

## Features

- Live video feed from webcam
- Real-time speed and acceleration monitoring
- Accident detection using motion analysis
- Visual and audio alerts
- Alert logging system
- User-friendly GUI interface

## Requirements

- Python 3.7 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone or download this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python accident_detection.py
```

2. The system will open a window showing:
   - Live video feed
   - Current speed
   - Current acceleration
   - System status
   - Alert button

3. The system will automatically:
   - Monitor motion and calculate speed/acceleration
   - Detect potential accidents based on sudden motion changes
   - Display alerts when accidents are detected
   - Log all alerts to 'accident_log.txt'

## How it Works

The system uses computer vision techniques to:
1. Capture live video feed from the webcam
2. Calculate motion between frames to estimate speed and acceleration
3. Detect sudden changes in motion that might indicate an accident
4. Provide visual and audio alerts when accidents are detected

## Notes

- The speed calculation is simplified and may need calibration for real-world use
- The accident detection threshold can be adjusted in the code
- Make sure your webcam is properly connected and accessible
- The system requires good lighting conditions for optimal performance 
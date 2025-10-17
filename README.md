# Drowsiness Detection System

A real-time computer vision system that monitors eye movements and facial features to detect drowsiness and alert the user. Perfect for preventing accidents during long drives or study sessions.

## Features

- Real-time eye tracking and monitoring
- Drowsiness detection using Eye Aspect Ratio (EAR)
- Yawn detection
- Audio alert system
- Blink counter
- Visual feedback with eye contours and status indicators
- Session summary statistics

## Requirements

- Python 3.7 - 3.11
- Webcam
- Windows OS (for audio alerts)

## Dependencies

```
opencv-python
mediapipe
numpy
```

## Installation

1. Clone or download the project

2. Install required packages:
```bash
pip install opencv-python mediapipe numpy
```

## Usage

Run the detection system:
```bash
python drowsiness_detection.py
```

## How It Works

The system uses MediaPipe Face Mesh to detect facial landmarks and calculates the Eye Aspect Ratio (EAR) to determine if eyes are open or closed.

### Eye Aspect Ratio (EAR)

EAR is calculated using the distances between specific eye landmarks:
- When eyes are open: EAR is higher
- When eyes are closed: EAR drops significantly
- If EAR stays below threshold for 20 consecutive frames, drowsiness alert is triggered

### Detection Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| EAR | 0.25 | Below this value indicates closed eyes |
| Consecutive Frames | 20 | Number of frames before drowsiness alert |
| Yawn Ratio | 0.6 | Mouth aspect ratio indicating yawning |

## Display Information

The system displays:
- **Status**: ACTIVE or DROWSY
- **EAR Value**: Current eye aspect ratio
- **Blinks**: Total number of blinks detected
- **Yawns**: Total number of yawns detected
- **EAR Bar**: Visual indicator of current EAR level
- **Eye Contours**: Green lines showing tracked eye regions

## Controls

| Key | Action |
|-----|--------|
| Q | Quit the application |
| R | Reset counters (blinks, yawns, drowsy frames) |

## Alerts

When drowsiness is detected:
- Red "DROWSINESS ALERT!" and "WAKE UP!" messages appear
- Audio beep plays (Windows only)
- Status changes to DROWSY

## Session Summary

Upon exit, the system displays:
- Total session duration
- Total blinks detected
- Total yawns detected
- Total drowsy frames

## Configuration

You can adjust detection sensitivity by modifying these parameters in the code:

```python
self.EAR_THRESHOLD = 0.25          # Lower = more sensitive
self.EAR_CONSEC_FRAMES = 20        # Frames before alert
self.YAWN_THRESHOLD = 0.6          # Yawn detection sensitivity
```

## Troubleshooting

**Camera not detected:**
- Ensure webcam is connected and not in use by another application
- Check camera permissions in system settings

**No face detected:**
- Ensure proper lighting
- Position face clearly in front of camera
- Keep entire face visible in frame

**False drowsiness alerts:**
- Adjust `EAR_THRESHOLD` to a lower value
- Increase `EAR_CONSEC_FRAMES` for longer detection window

**Audio alerts not working:**
- System uses Windows `winsound` library
- On Mac/Linux, alerts will print to console instead

## Technical Details

- Framework: MediaPipe Face Mesh
- Face Landmarks: 468 landmarks per face
- Eye Landmarks: 6 points per eye
- Processing: Real-time at camera frame rate
- Resolution: 640x480 default

## Use Cases

- Driver drowsiness monitoring
- Student focus monitoring during study
- Night shift worker alertness
- Long meeting participant engagement
- Any scenario requiring alertness monitoring

## Limitations

- Requires good lighting conditions
- Face must be clearly visible
- Works best with frontal face view
- Glasses may affect accuracy
- Audio alerts Windows-only

## Author

Created by Adnan
projects/adnan.pkg

## License

This project is open source and available for educational and personal use.

## Acknowledgments

- MediaPipe for face mesh detection
- OpenCV for computer vision processing

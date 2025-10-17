import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 20
        self.YAWN_THRESHOLD = 0.6
        
        self.eye_closed_counter = 0
        self.total_blinks = 0
        self.yawn_counter = 0
        self.drowsy_frames = 0
        
        self.is_drowsy = False
        self.alarm_on = False
        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [61, 291, 0, 17, 269, 405]
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        A = self.euclidean_distance(eye_landmarks[1], eye_landmarks[5])
        B = self.euclidean_distance(eye_landmarks[2], eye_landmarks[4])
        C = self.euclidean_distance(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks):
        A = self.euclidean_distance(mouth_landmarks[1], mouth_landmarks[5])
        B = self.euclidean_distance(mouth_landmarks[0], mouth_landmarks[3])
        mar = A / B
        return mar
    
    def get_landmarks(self, landmarks, indices, frame_width, frame_height):
        coords = []
        for idx in indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            coords.append((x, y))
        return coords
    
    def play_alarm(self):
        try:
            winsound.Beep(1000, 200)
        except:
            print("ALERT: DROWSINESS DETECTED!")
    
    def draw_eye_contour(self, frame, eye_coords):
        for i in range(len(eye_coords)):
            cv2.line(frame, eye_coords[i], eye_coords[(i + 1) % len(eye_coords)], 
                    (0, 255, 0), 1)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Drowsiness Detection System Started!")
        print("- System will alert if eyes are closed for too long")
        print("- Monitors yawning and eye closure")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset counters")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            frame_height, frame_width = frame.shape[:2]
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_coords = self.get_landmarks(
                        face_landmarks.landmark, self.LEFT_EYE, frame_width, frame_height
                    )
                    right_eye_coords = self.get_landmarks(
                        face_landmarks.landmark, self.RIGHT_EYE, frame_width, frame_height
                    )
                    mouth_coords = self.get_landmarks(
                        face_landmarks.landmark, self.MOUTH, frame_width, frame_height
                    )
                    
                    left_ear = self.calculate_eye_aspect_ratio(left_eye_coords)
                    right_ear = self.calculate_eye_aspect_ratio(right_eye_coords)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    mar = self.calculate_mouth_aspect_ratio(mouth_coords)
                    
                    self.draw_eye_contour(frame, left_eye_coords)
                    self.draw_eye_contour(frame, right_eye_coords)
                    
                    if avg_ear < self.EAR_THRESHOLD:
                        self.eye_closed_counter += 1
                        
                        if self.eye_closed_counter >= self.EAR_CONSEC_FRAMES:
                            self.is_drowsy = True
                            self.drowsy_frames += 1
                            
                            if not self.alarm_on:
                                self.alarm_on = True
                                self.play_alarm()
                            
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                            cv2.putText(frame, "WAKE UP!", (10, 150),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    else:
                        if self.eye_closed_counter >= self.EAR_CONSEC_FRAMES:
                            self.total_blinks += 1
                        
                        self.eye_closed_counter = 0
                        self.is_drowsy = False
                        self.alarm_on = False
                    
                    if mar > self.YAWN_THRESHOLD:
                        self.yawn_counter += 1
                        cv2.putText(frame, "YAWNING DETECTED", (10, 200),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    status = "DROWSY" if self.is_drowsy else "ACTIVE"
                    status_color = (0, 0, 255) if self.is_drowsy else (0, 255, 0)
                    
                    cv2.putText(frame, f"Status: {status}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Yawns: {self.yawn_counter}", (400, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    bar_length = int(avg_ear * 200)
                    bar_color = (0, 255, 0) if avg_ear > self.EAR_THRESHOLD else (0, 0, 255)
                    cv2.rectangle(frame, (400, 60), (400 + bar_length, 80), bar_color, -1)
                    cv2.rectangle(frame, (400, 60), (600, 80), (255, 255, 255), 2)
                    
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit | 'r' to reset", 
                       (10, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Drowsiness Detection System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.total_blinks = 0
                self.yawn_counter = 0
                self.drowsy_frames = 0
                print("Counters reset!")
        
        elapsed_time = time.time() - start_time
        print("\n=== Session Summary ===")
        print(f"Duration: {elapsed_time:.2f} seconds")
        print(f"Total Blinks: {self.total_blinks}")
        print(f"Total Yawns: {self.yawn_counter}")
        print(f"Drowsy Frames: {self.drowsy_frames}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()
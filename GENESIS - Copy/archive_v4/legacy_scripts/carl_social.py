"""
carl_social.py — CARL's Asynchronous 10Hz Social Visual Cortex (Phase C1).

Architecture:
- SocialVisualLobe: Runs a background daemon thread that captures camera frames via
  OpenCV and extracts scale-invariant human facial expressions via MediaPipe Face Landmarker at ~10Hz.
- Zero-Blocking Interface: The thread writes a 5-element float vector to CARL's SensorStateBuffer:
  `[face_present, smile_ratio, eye_opening, eyebrow_raise, face_distance]`
- Decoupled execution: Frame read blocks (up to 30ms) occur entirely in the background thread,
  safeguarding the 100Hz real-time control tick and 50Hz RL planning threads from latency spikes.
- Graceful Fallback: Auto-detects missing packages, locked cameras, or dark frames, transitioning
  silently to an all-zero neutral state without causing training crashes.
"""

import sys
import os
import time
import math
import numpy as np
import threading

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python.vision import face_landmarker as fl
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    OPENCV_MEDIAPIPE_AVAILABLE = True
except ImportError:
    OPENCV_MEDIAPIPE_AVAILABLE = False


class SocialVisualLobe:
    def __init__(self, shared_sensor_buffer=None, camera_index=0, frame_rate=10):
        self.sensor_buffer = shared_sensor_buffer
        self.camera_index = camera_index
        self.delay_seconds = 1.0 / frame_rate
        
        self._running = False
        self._thread = None
        self._latest_vector = np.zeros(5, dtype=np.float32) # [present, smile, eye_opening, eyebrow_raise, distance]
        self._vector_lock = threading.Lock()
        
        # Absolute path to the model file to ensure reliable loading in any execution context
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
        
    def start(self):
        """Spins up the background visual processing thread."""
        if not OPENCV_MEDIAPIPE_AVAILABLE:
            print("[SOCIAL] MediaPipe, OpenCV, or tasks submodules are missing. Running in fallback (null-vector) mode.")
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="SocialVisualCortex", daemon=True)
        self._thread.start()
        print("[SOCIAL] Asynchronous 10Hz Social Visual thread started.")

    def stop(self):
        """Gracefully halts the capture loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("[SOCIAL] Visual thread stopped cleanly.")

    def get_latest_face_vector(self):
        """Thread-safe local access to raw visual data."""
        with self._vector_lock:
            return self._latest_vector.copy()

    def _dist(self, p1, p2):
        """Euclidean distance helper for 3D landmarks."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _capture_loop(self):
        """Decoupled 10Hz webcam ingestion and Face Landmarker processing loop."""
        # Initialize OpenCV Camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[SOCIAL] [WARNING] Could not open camera source at index {self.camera_index}. Entering fallback mode.")
            self._write_null_vector()
            
            # Loop silently to handle hot-plugging attempts later
            while self._running:
                time.sleep(self.delay_seconds)
            return

        # Disable autofocus latency if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize MediaPipe Face Landmarker
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = fl.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionTaskRunningMode.IMAGE,
                num_faces=1
            )
            detector = fl.FaceLandmarker.create_from_options(options)
            print(f"[SOCIAL] FaceLandmarker initialized from {self.model_path}")
        except Exception as e:
            print(f"[SOCIAL] [ERROR] Failed to load FaceLandmarker: {e}. Entering fallback mode.")
            self._write_null_vector()
            cap.release()
            while self._running:
                time.sleep(self.delay_seconds)
            return

        print("[SOCIAL] Camera initialized. Capturing frames...")

        while self._running:
            t_start = time.perf_counter()
            
            # OpenCV Read frame (blocks for up to ~30ms depending on webcam driver)
            ret, frame = cap.read()
            if not ret or frame is None:
                self._write_null_vector()
                time.sleep(self.delay_seconds)
                continue

            # Convert BGR frame to RGB for MediaPipe inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            try:
                results = detector.detect(mp_image)
            except Exception as e:
                print(f"[SOCIAL] Inference error: {e}")
                self._write_null_vector()
                time.sleep(self.delay_seconds)
                continue

            face_present = 0.0
            smile = 0.0
            eye_opening = 0.5
            eyebrow_raise = 0.5
            distance = 0.0

            if results.face_landmarks:
                face_landmarks = results.face_landmarks[0]
                face_present = 1.0

                try:
                    # ── Scale Normalization: Pupil Distance (Stable Metric) ──
                    num_landmarks = len(face_landmarks)
                    if num_landmarks >= 478:
                        # Pupil Landmark mappings: Left pupil (468), Right pupil (473)
                        pupil_L = face_landmarks[468]
                        pupil_R = face_landmarks[473]
                    else:
                        # Fallback: left eye corners (33, 133) and right eye corners (263, 362)
                        p1_L, p2_L = face_landmarks[33], face_landmarks[133]
                        p1_R, p2_R = face_landmarks[263], face_landmarks[362]
                        class DummyPoint:
                            def __init__(self, x, y):
                                self.x = x
                                self.y = y
                        pupil_L = DummyPoint((p1_L.x + p2_L.x)/2.0, (p1_L.y + p2_L.y)/2.0)
                        pupil_R = DummyPoint((p1_R.x + p2_R.x)/2.0, (p1_R.y + p2_R.y)/2.0)

                    pupil_dist = self._dist(pupil_L, pupil_R)
                    
                    if pupil_dist > 1e-4:
                        # 1. Smile Ratio: Distance between mouth corners (61, 291)
                        mouth_L = face_landmarks[61]
                        mouth_R = face_landmarks[291]
                        mouth_width = self._dist(mouth_L, mouth_R)
                        # Normal resting width is ~0.45*pupil_dist. Full smile is ~0.70*pupil_dist
                        smile = np.clip((mouth_width / pupil_dist - 0.45) / 0.25, 0.0, 1.0)

                        # 2. Eye Opening: Average vertical eyelid openings
                        # Left Eye: top (159) to bottom (145)
                        # Right Eye: top (386) to bottom (374)
                        eye_L = self._dist(face_landmarks[159], face_landmarks[145])
                        eye_R = self._dist(face_landmarks[386], face_landmarks[374])
                        avg_eye_height = (eye_L + eye_R) / 2.0
                        # Resting height is ~0.10*pupil_dist. Squint/Blink is ~0.0. Wide open is ~0.18*pupil_dist
                        eye_opening = np.clip(avg_eye_height / (0.18 * pupil_dist), 0.0, 1.0)

                        # 3. Eyebrow Raise: Distance between eyebrow peaks and pupil centers
                        # Left Eyebrow Peak (105) to Left Pupil (468/or eye center)
                        # Right Eyebrow Peak (334) to Right Pupil (473/or eye center)
                        brow_L = self._dist(face_landmarks[105], pupil_L)
                        brow_R = self._dist(face_landmarks[334], pupil_R)
                        avg_brow_dist = (brow_L + brow_R) / 2.0
                        # Normal is ~0.30*pupil_dist. raised is ~0.45*pupil_dist, furrowed is ~0.22*pupil_dist
                        brow_ratio = avg_brow_dist / pupil_dist
                        eyebrow_raise = np.clip((brow_ratio - 0.22) / 0.20, 0.0, 1.0)

                        # 4. Face Proximity / Distance (Pupil size mapping)
                        # Close face = large pupil distance (up to 0.3 of frame width). Far face = small (0.05)
                        distance = np.clip(pupil_dist / 0.25, 0.0, 1.0)
                except IndexError:
                    pass # Catch occasional landmark index drops

            # Compile into 5D state array
            latest = np.array([face_present, smile, eye_opening, eyebrow_raise, distance], dtype=np.float32)
            
            with self._vector_lock:
                self._latest_vector = latest

            # Write atomically to shared buffer if hooked
            if self.sensor_buffer is not None:
                self.sensor_buffer.write_face(latest)

            # Enforce 10Hz framerate sleep
            elapsed = time.perf_counter() - t_start
            rem = self.delay_seconds - elapsed
            if rem > 0:
                time.sleep(rem)

        # Release resources
        cap.release()
        try:
            detector.close()
        except Exception:
            pass

    def _write_null_vector(self):
        """Pushes an all-zero neutral vector when visual processing is dead."""
        null_vec = np.zeros(5, dtype=np.float32)
        with self._vector_lock:
            self._latest_vector = null_vec
        if self.sensor_buffer is not None:
            self.sensor_buffer.write_face(null_vec)


# ═════════════════════════════════════════════════════════════════════════════
#  STANDALONE VISUALIZATION & DIAGNOSTICS TEST SUITE
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("  carl_social.py -- Asynchronous Social Visual Cortex Validation")
    print("  (Opens camera and renders detected metrics to console at 2Hz)")
    print("  Press Ctrl+C to stop.")
    print("=" * 72)

    social = SocialVisualLobe(camera_index=0, frame_rate=10)
    social.start()

    try:
        while True:
            vec = social.get_latest_face_vector()
            print(f"Face Present: {int(vec[0])} | "
                  f"Smile: {vec[1]*100:>3.0f}% | "
                  f"Eye Open: {vec[2]*100:>3.0f}% | "
                  f"Brow Raise: {vec[3]*100:>3.0f}% | "
                  f"Distance: {vec[4]*100:>3.0f}%")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping visual lobe...")
        social.stop()
        print("Done.")

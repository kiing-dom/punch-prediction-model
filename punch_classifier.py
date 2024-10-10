import cv2
import mediapipe as mp
import numpy as np
import joblib

# loading the model and the scaler
model = joblib.load('models/best_punch_prediction_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# initialize the camera
cap = cv2.VideoCapture(0)

def extract_features(landmarks):
    # Extract relevant landmarks and compute angles
    features = []
    for landmark in [landmarks.left_shoulder, landmarks.right_shoulder, 
                     landmarks.left_elbow, landmarks.right_elbow,
                     landmarks.left_wrist, landmarks.right_wrist,
                     landmarks.left_hip, landmarks.right_hip]:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    left_elbow_angle = compute_angle(landmarks.left_shoulder, landmarks.left_elbow, landmarks.left_wrist)
    right_elbow_angle = compute_angle(landmarks.right_shoulder, landmarks.right_elbow, landmarks.right_wrist)
    features.extend([left_elbow_angle, right_elbow_angle])
    
    return np.array(features).reshape(1, -1)

def compute_angle(a, b, c):
    # Compute the angle between three points
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def classify_punch(frame):

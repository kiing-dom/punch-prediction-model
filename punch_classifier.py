import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

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
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    if results.pose_landmarks:
        features = extract_features(results.pose_landmarks.LANDMARKS)
        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

        return prediction, probabilities
    
    return None, None


class PunchPredictor:
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.punch_history = deque(maxlen=sequence_length)
        self.transition_matrix = {}

    def update(self, punch):
        if len(self.punch_history) == self.sequence_length:
            sequence = tuple(self.punch_history)
            if sequence not in self.transition_matrix:
                self.transition_matrix[sequence] = {}
            if punch not in self.transition_matrix[sequence]:
                self.transition_matrix[sequence][punch] = 0
            self.transition_matrix[sequence][punch] += 1

        self.punch_history.append(punch)
    
    def predict_next(self, top_n=3):
        if len(self.punch_history) < self.sequence_length:
            return []

        sequence = tuple(self.punch_history)
        if sequence in self.transition_matrix:
            predictions = sorted(self.transition_matrix[sequence].items(),
                                 key = lambda x: x[1], reverse=True)
            return [p[0] for p in predictions[:top_n]]
        return []

punch_predictor = PunchPredictor()

def main():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # classify the current punch
        punch, _ = classify_punch(frame)

        if punch is not None:
            # Updating the punch predictor
            punch_predictor.update(punch)

            # Get the 3 most likely Next Punches
            next_punches = punch_predictor.predict_next(top_n=3)

            # Display the Results on the UI
            cv2.putText(frame, f"Current: {punch}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            for i, next_punch in enumerate(next_punches):
                cv2.putText(frame, f"Next {i + 1}: {next_punch}", (10, 60 + 30*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Punch Classifier', frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
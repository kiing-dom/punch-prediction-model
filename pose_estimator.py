import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to process video frames
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #process the frame and get the pose landmarks
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
    return frame

    # Capture video from camera or file input

cap = cv2.VideoCapture(0)

desired_width = 1280
desired_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

cv2.namedWindow('Boxing Pose Estimator', cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    annotated_frame = process_frame(frame)
    
    cv2.imshow('Pose Estimation', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
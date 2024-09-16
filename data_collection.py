import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime
from pose_estimator import pose
from data_collection import process_frame

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode = False,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process_frame(rgb_frame)
    return results

def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y. lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
    return pose

def main():
    cap = cv2.VideoCapture(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"boxing_data_{timestamp}.csv"

    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        header = ['timestamp'] + [f'keypoint_{i}_coord' for i in range(33) for coord in ['x', 'y', 'z', 'visibility']]
        csv_writer.writerow(header)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            results = process_frame(frame)

            pose_keypoints = extract_keypoints(results)

            #Log data
            

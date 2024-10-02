import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime
import argparse

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LANDMARKS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24
}

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    return results

def extract_keypoints(results):
    if results.pose_landmarks:
        keypoints = {}
        for name, index in LANDMARKS.items():
            lm = results.pose_landmarks.landmark[index]
            keypoints[name] = f"{lm.x:.4f},{lm.y:.4f},{lm.z:.4f}"
    else:
        keypoints = {name:"0,0,0" for name in LANDMARKS.keys()}
    return keypoints

def process_video(video_path, movement_type, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Detected FPS: {fps}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"boxing_data_{movement_type}_{timestamp}.csv")
    
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        header = ['frame', 'timestamp', 'movement_type'] + list(LANDMARKS.keys())
        csv_writer.writerow(header)
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            results = process_frame(frame)
            keypoints = extract_keypoints(results)
            
            timestamp = frame_count / fps
            row = [frame_count, f"{timestamp:.3f}", movement_type] + list(keypoints.values())
            csv_writer.writerow(row)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
    
    cap.release()
    print(f"Finished processing {video_path}. Output saved to {csv_filename}")

def main():
    parser = argparse.ArgumentParser(description='Process boxing videos and extract pose data')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('movement_type', type=str, help='Type of movement in the video (e.g., jab, cross, hook)')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save output CSV files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    process_video(args.video_path, args.movement_type, args.output_dir)

if __name__ == "__main__":
    main()
import os
import cv2
import numpy as np
import csv
from handshape_feature_extractor import HandShapeFeatureExtractor

# Mapping dictionaries for test and training videos.
test_gesture_mapping = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "DecreaseFanSpeed": 10,
    "FanOff": 11,
    "FanOn": 12,
    "IncreaseFanSpeed": 13,
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16
}

training_gesture_mapping = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Decrease_fan_speed": 10,
    "Turn_off_fan": 11,
    "Turn_on_fan": 12,
    "Increase_fan_speed": 13,
    "Turn_off_lights": 14,
    "Turn_on_lights": 15,
    "Set_Thermostat": 16
}

def extract_middle_frame(video_path):
    """
    Extracts the middle frame of a given video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # print("Error opening video:", video_path)
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        # print("Error reading frame from:", video_path)
        return None

def get_penultimate_features(folder):
    # =============================================================================
    # Get the penultimate layer for the given folder (training or test data)
    # =============================================================================
    # Extract the middle frame of each gesture video
    extractor = HandShapeFeatureExtractor.get_instance()
    features = []
    labels = []
    
    video_files = sorted([f for f in os.listdir(folder) if f.endswith('.mp4')])
    
    for video_file in video_files:
        video_path = os.path.join(folder, video_file)
        frame = extract_middle_frame(video_path)
        if frame is not None:
            # Convert frame to grayscale to match expected input shape
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            feature_vector = extractor.extract_feature(gray_frame)
            features.append(feature_vector)
            
            # Extract gesture name from the filename (filename without extension)
            base_name = os.path.splitext(video_file)[0]
            if folder == "traindata":
                # For training videos, assume the gesture name is before the delimiter "_PRACTICE_"
                gesture_name = base_name.split('_PRACTICE_')[0].strip()
                if gesture_name in training_gesture_mapping:
                    labels.append(training_gesture_mapping[gesture_name])
                # else:
                    # print("Unrecognized gesture name in training video filename:", video_file)
            elif folder == "test":
                # For test videos, we extract gesture name based on a hyphen delimiter
                if '-' in base_name:
                    gesture_name = base_name.split('-')[-1].strip()
                else:
                    gesture_name = base_name
                labels.append(gesture_name)
    return features, labels

def compute_cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two feature vectors.
    """
    v1 = vec1.flatten()
    v2 = vec2.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def recognize_gestures(test_features, train_features, train_labels):
    # =============================================================================
    # Recognize the gesture (use cosine similarity for comparing the vectors)
    # =============================================================================
    recognized_labels = []
    for test_vec in test_features:
        similarities = []
        for train_vec in train_features:
            sim = compute_cosine_similarity(test_vec, train_vec)
            similarities.append(sim)
        best_match_index = np.argmax(similarities)
        recognized_labels.append(train_labels[best_match_index])
    return recognized_labels

def main():
    # Define folders for training and test videos
    train_folder = "traindata"
    test_folder = "test"
    
    # Get penultimate layer features for training and test data
    train_features, train_labels = get_penultimate_features(train_folder)
    test_features, test_video_names = get_penultimate_features(test_folder)
    
    # Recognize gestures in test data
    predictions = recognize_gestures(test_features, train_features, train_labels)
    
    # Write the predictions to "Results.csv" as a 51 x 1 matrix (no header, only predicted labels)
    with open("Results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(51):
        # for pred in predictions:
            writer.writerow([predictions[i]])

if __name__ == "__main__":
    main()

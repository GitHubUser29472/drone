import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from codrone_edu.drone import Drone
import time

# Load MoveNet model using TensorFlow Hub (singlepose lightning model)
model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(model_url)

# Access the model's 'signatures' for inference
movenet = model.signatures['serving_default']

# Initialize the drone
drone = Drone()
drone.pair()
print("Paired!")

# Capture video from the drone camera or webcam
cap = cv2.VideoCapture(0)  # Replace with your drone camera feed if necessary

# Function to run MoveNet pose detection on a frame
def detect_pose(frame):
    # Resize the frame to 192x192 for MoveNet model input
    input_frame = cv2.resize(frame, (192, 192))
    
    # Convert the frame to a TensorFlow tensor and change dtype to int32
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.int32)  # Change to int32
    
    # Convert it to a tensor
    input_frame = tf.convert_to_tensor(input_frame)
    
    # Run inference with the model (use the 'serving_default' signature)
    output = movenet(input_frame)

    # Extract keypoints from the output
    keypoints = output['output_0'].numpy().squeeze()
    return keypoints

# Function to visualize and debug keypoints
def draw_keypoints(frame, keypoints):
    for i, keypoint in enumerate(keypoints):
        x, y, confidence = keypoint
        if confidence > 0.1:  # Show all keypoints with at least 0.1 confidence
            # Draw a circle at each keypoint for better visibility
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)
    return frame

# Variable to track the drone's air status
drone_in_air = False
prev_arm_y = None  # Variable to track the previous arm position

# Function to detect arm position and control drone (takeoff/land)
def detect_arm_and_control(keypoints):
    global drone_in_air, prev_arm_y

    # Get the keypoints for the left arm (elbow and wrist)
    left_elbow = keypoints[7]  # Left elbow
    left_wrist = keypoints[9]  # Left wrist
    
    # Extract coordinates and confidence from keypoints
    left_elbow_x, left_elbow_y, left_elbow_conf = left_elbow
    left_wrist_x, left_wrist_y, left_wrist_conf = left_wrist

    # Debugging output for keypoints
    print(f"Left Elbow: {left_elbow}")
    print(f"Left Wrist: {left_wrist}")

    # Ensure the wrist or elbow is visible (confidence > 0.1)
    if left_elbow_conf > 0.1 and left_wrist_conf > 0.1:
        # Choose wrist or elbow, we are using wrist for y-coordinate
        current_arm_y = left_wrist_y  # or you can use left_elbow_y

        # Check if the arm has moved up or down compared to the previous position
        if prev_arm_y is not None:
            if current_arm_y < prev_arm_y - 0.05:  # Arm moved up
                if not drone_in_air:
                    # Takeoff if not already in air
                    drone.takeoff()  # Use takeoff() to start flying
                    drone_in_air = True
                    print("Takeoff initiated!")
            elif current_arm_y > prev_arm_y + 0.05:  # Arm moved down
                if drone_in_air:
                    # Land if drone is in the air
                    drone.land()  # Use land() to bring the drone down
                    drone_in_air = False
                    print("Landing initiated!")
        
        # Update the previous arm position for the next iteration
        prev_arm_y = current_arm_y

# Main loop for pose detection and drone control
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Flip the frame to make it a mirror image
    frame = cv2.flip(frame, 1)

    # Detect pose keypoints using MoveNet
    keypoints = detect_pose(frame)

    # Draw keypoints on the frame (for visualization)
    frame = draw_keypoints(frame, keypoints)

    # Detect arm position and control the drone
    detect_arm_and_control(keypoints)

    # Display the resulting frame with pose detection
    cv2.imshow('Pose Detection with MoveNet', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

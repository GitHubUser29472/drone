import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from codrone_edu.drone import Drone

# Load MoveNet model using TensorFlow Hub
model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(model_url)

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
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)
    input_frame = input_frame / 255.0  # Normalize image

    # Run inference with the model
    output = model(input_frame) 

    # Extract keypoints from the output
    keypoints = output['output_0'].numpy().squeeze()
    return keypoints

# Variables to track previous pose positions
prev_keypoints = None

motion_threshold = 0.05  # Threshold to trigger drone actions

print("Starting pose detection...")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to make it a mirror image
    frame = cv2.flip(frame, 1)

    # Detect pose keypoints using MoveNet
    keypoints = detect_pose(frame)

    # Draw keypoints and skeleton on the frame
    for i, keypoint in enumerate(keypoints):
        x, y, confidence = keypoint
        if confidence > 0.1:  # Only consider high-confidence keypoints
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)

    # Compare current keypoints with previous ones to detect motion
    if prev_keypoints is not None:
        significant_motion_detected = False
        for i in range(len(keypoints)):
            x, y, _ = keypoints[i]
            prev_x, prev_y, _ = prev_keypoints[i]

            if np.abs(x - prev_x) > motion_threshold or np.abs(y - prev_y) > motion_threshold:
                print(f"Keypoint {i} moved significantly!")
                significant_motion_detected = True
                break  # Exit loop after detecting motion

        # Trigger drone action if significant motion is detected
        if significant_motion_detected:
            if not drone.is_in_air():
                drone.takeoff()
                drone.up()
                print("Motion detected - Drone in the air!")
            else:
                drone.forward()
                print("Moving forward in response to motion!")

    # Store the current keypoints for the next iteration
    prev_keypoints = keypoints

    # Display the resulting frame with pose detection
    cv2.imshow('Pose Detection with MoveNet', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()





# drone = Drone()
# drone.pair()
# print("Paired!")
# drone.takeoff()
# drone.up()
# drone.flip()
# print("In the air!")
# print("Landing")
# drone.land()
# drone.close()
# print("Program complete")


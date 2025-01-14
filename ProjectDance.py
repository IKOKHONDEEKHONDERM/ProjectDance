import cv2
import mediapipe as mp
import pygame

# Initialize Mediapipe and Pygame
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pygame.mixer.init()

# Load sounds for each body part
sounds = {
    "head": pygame.mixer.Sound("D:\WORK\Year-3-2\design\DO.wav"),
    "left_shoulder": pygame.mixer.Sound("D:\WORK\Year-3-2\design\RE.wav"),
    "right_shoulder": pygame.mixer.Sound("D:\WORK\Year-3-2\design\MI.wav"),
    "chest": pygame.mixer.Sound("D:\WORK\Year-3-2\design\FA.wav"),
    "stomach": pygame.mixer.Sound("D:\WORK\Year-3-2\design\SOL.wav")
}

# Set up webcam
cap = cv2.VideoCapture(0)
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = pose.process(rgb_frame)

        # Draw body part rectangles and play sounds
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Map landmarks to body parts
            body_parts = {
                "head": landmarks[0],  # Nose
                "left_shoulder": landmarks[11],  # Left shoulder
                "right_shoulder": landmarks[12],  # Right shoulder
                "chest": landmarks[11],  # Use left shoulder as reference for chest
                "stomach": landmarks[23]  # Mid-hip
            }

            for part, landmark in body_parts.items():
                x, y = int(landmark.x * screen_width), int(landmark.y * screen_height)
                width, height = 100, 100  # Dimensions of the rectangle

                # Draw rectangle around the body part
                cv2.rectangle(frame, (x - width // 2, y - height // 2), (x + width // 2, y + height // 2), (255, 0, 0), 2)
                cv2.putText(frame, part, (x - 40, y - height // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Play sound if part is detected
                sounds[part].play()

        # Display the frame
        cv2.imshow('Body Part Detection', frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
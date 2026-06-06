import cv2
import mediapipe as mp
import numpy as npQ
import collections
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create deques to store position history for movement detection
position_history = collections.deque(maxlen=15)
hand_position_history = collections.deque(maxlen=15)
punch_history = collections.deque(maxlen=10)

# Alert system variables
alerted_activities = set()  # Set to keep track of activities that have already been alerted
alertable_activities = ["Boxing", "Running"]

def send_email_alert(activity, confidence, recipient_email):
    """Send email alert when anomalous activity is detected"""
    # Skip if we've already alerted for this activity in this session
    if activity in alerted_activities:
        return False
    
    # Add to alerted activities
    alerted_activities.add(activity)
    
    # Email configuration
    sender_email = os.environ.get("ALERT_EMAIL", "maheshbabu77k2@gmail.com")
    sender_password = os.environ.get("ALERT_PASSWORD", "nnpp jskv ypjz yyfj")
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"ANOMALY ALERT: {activity} Activity Detected"
    
    # Email body
    body = f"""
    Anomaly Detection Alert
    
    Anomalous Activity: {activity}
    Confidence: {confidence:.2f}
    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    This is an automated alert from your Activity Recognition System.
    This email is sent as a one-time notification for this detected activity.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # Send email
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"Anomaly alert email sent for {activity}")
        return True
    except Exception as e:
        print(f"Failed to send email alert: {e}")
        return False

def calculate_movement(current_positions, history):
    """Calculate movement from position history"""
    if len(history) < 2:
        return 0
    movement = sum(np.mean(np.abs(history[i] - history[i+1])) for i in range(len(history) - 1))
    return movement

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def detect_activity(landmarks):
    """Enhanced activity detection with missing landmark handling"""

    required_landmarks = [
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    
    # Check if key lower-body landmarks are visible
    visible_landmarks = [landmarks.landmark[l] for l in required_landmarks if landmarks.landmark[l].visibility > 0.5]
    
    if len(visible_landmarks) < len(required_landmarks):
        return "No Activity Detected", 0.0  # If lower body isn't visible, return early

    # Extract key points
    left_hip, right_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee, right_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle, right_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_wrist, right_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder, right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Store positions for movement detection
    current_positions = np.array([[left_hip.x, left_hip.y], [right_hip.x, right_hip.y],
                                  [left_knee.x, left_knee.y], [right_knee.x, right_knee.y],
                                  [left_ankle.x, left_ankle.y], [right_ankle.x, right_ankle.y]])
    position_history.append(current_positions)

    hand_positions = np.array([[left_wrist.x, left_wrist.y], [right_wrist.x, right_wrist.y]])
    hand_position_history.append(hand_positions)

    # Calculate movements
    body_movement = calculate_movement(current_positions, position_history)
    hand_movement = calculate_movement(hand_positions, hand_position_history)

    # Calculate angles and distances
    hip_knee_angle_left = calculate_angle(left_hip, left_knee, left_ankle)
    hip_knee_angle_right = calculate_angle(right_hip, right_knee, right_ankle)
    wrist_distance = np.linalg.norm(np.array([left_wrist.x, left_wrist.y]) - np.array([right_wrist.x, right_wrist.y]))
    knee_distance = np.linalg.norm(np.array([left_knee.x, left_knee.y]) - np.array([right_knee.x, right_knee.y]))

    # Sitting detection
    hip_height = (left_hip.y + right_hip.y) / 2
    knee_height = (left_knee.y + right_knee.y) / 2
    is_sitting = (hip_knee_angle_left < 120 and hip_knee_angle_right < 120 and hip_height > knee_height and knee_height > 0.6)

    # Boxing detection
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    left_punch = abs(left_wrist.x - left_shoulder.x) > shoulder_width * 1.2
    right_punch = abs(right_wrist.x - right_shoulder.x) > shoulder_width * 1.2
    punch_history.append(left_punch or right_punch)
    is_boxing = sum(punch_history) >= 3 and hand_movement > 0.15

    # Running detection
    is_running = (body_movement > 0.15 and knee_distance > 0.25)

    # Activity classification
    if is_sitting:
        return "Sitting", min(1.0, (120 - hip_knee_angle_left) / 60)
    elif is_boxing:
        return "Boxing", min(1.0, hand_movement * 5)
    elif is_running:
        return "Running", min(1.0, body_movement * 4)
    elif wrist_distance < 0.15 and hand_movement > 0.1:
        return "Clapping", min(1.0, (0.15 - wrist_distance) * 10)
    elif body_movement > 0.1 and knee_distance > 0.2:
        return "Walking", min(1.0, body_movement * 3)
    else:
        return "Standing", max(0.3, 1.0 - body_movement * 2)

def get_activity_color(activity):
    """Return BGR color code for each activity"""
    return {
        "Boxing": (0, 0, 255),
        "Running": (0, 255, 0),
        "Walking": (255, 165, 0),
        "Sitting": (255, 0, 255),
        "Clapping": (255, 255, 0),
        "Standing": (255, 255, 255),
        "No Activity Detected": (0, 0, 0)  # Black for no detection
    }.get(activity, (255, 255, 255))

def main():
    """Main function to process webcam feed"""
    print("Initializing webcam...")
    
    # Get email from command line or use default
    recipient_email = input("Enter email address for alerts: ")
    if not recipient_email or '@' not in recipient_email:
        print("Invalid email address. Using default: baskaranv850@gmail.com")
        recipient_email = "baskaranv850@gmail.com"
    
    # Set activity confidence threshold for alerts
    alert_threshold = 0.7
    
    # Initialize webcam
    cap = None
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
    if cap is None or not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully! Press 'q' to quit")
    print(f"Monitoring for anomalous activities: {', '.join(alertable_activities)}")
    print(f"Alerts will be sent to: {recipient_email}")
    print("Note: Each anomalous activity will trigger only ONE email alert per session")
    
    prev_time = time.time()
    activity_durations = {}  # Track how long activities have been occurring

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            activity, confidence = detect_activity(results.pose_landmarks)
            activity_color = get_activity_color(activity)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display activity
            cv2.putText(frame, f'Activity: {activity}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, activity_color, 2)
            cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, activity_color, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Anomaly detection for activities we care about
            if activity in alertable_activities and confidence > alert_threshold:
                # Initialize or update activity duration
                if activity not in activity_durations:
                    activity_durations[activity] = 0
                activity_durations[activity] += 1
                
                # Send alert if activity continues for several consecutive frames
                if activity_durations[activity] > 30 and activity not in alerted_activities:
                    # Display alert status
                    alert_text = "ANOMALY DETECTED!"
                    cv2.putText(frame, alert_text, (frame.shape[1] - 300, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Send email alert (only once per activity)
                    send_email_alert(activity, confidence, recipient_email)
            else:
                # Reset duration if activity not detected
                if activity in activity_durations:
                    activity_durations[activity] = 0
            
            # Display alert status for any activities we've already alerted on
            y_offset = 120
            for act in alertable_activities:
                if act in alerted_activities:
                    status_text = f"{act}: Alert Sent"
                    cv2.putText(frame, status_text, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30

        cv2.imshow('Activity Recognition with Anomaly Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

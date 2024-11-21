import cv2
import math
import numpy as np
import threading
from pynput import keyboard
from ultralytics import YOLO

class AdvancedObjectDetection:
    def __init__(self, model_path="yolov8n.pt"):
        print("Loading YOLOv8 for Object Detection")
        self.model = YOLO(model_path)  # Load the YOLOv8 model
        self.classes = self.model.names  # Class names from the model
        self.conf_threshold = 0.5  # Confidence threshold
        self.nms_threshold = 0.4  # NMS threshold

    def detect(self, frame):
        # Perform detection on the frame
        results = self.model.predict(frame, conf=self.conf_threshold, iou=self.nms_threshold, device="cpu")
        return results

# Global flags and variables
f_pressed = False
s_pressed = False
exit_flag = False
mouse_x, mouse_y = -1, -1
selected_object = None  # Holds the selected object (box)
selected_center = None  # Holds the center of the selected object (cx, cy)
selected_class = None  # Holds the class name of the selected object
trajectory = []  # List to store trajectory points
show_trajectory = False  # Toggle to show/hide trajectory

def on_press(key):
    global f_pressed, s_pressed, exit_flag
    if key == keyboard.Key.esc: 
        exit_flag = True
        return False
    try:
        key = key.char
    except AttributeError:
        print(f"\n\nERROR: Special key {key} pressed\n\n")
        exit_flag = True
        return False
    if key == "f":
        f_pressed = True
    if key == "s":
        s_pressed = True
    if key == "esc":  # Allow exiting with ESC key
        exit_flag = True


def mouse_callback(event, x, y, flags, param):
    """
    Tracks the mouse coordinates.
    """
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def calculate_center(box):
    """Calculate the center of a bounding box."""
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


def euclidean_distance(pt1, pt2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


# Function to start the listener in a separate thread
def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# Start the listener
key_thread = threading.Thread(target=start_listener, daemon=True)
key_thread.start()

# Initialize Object Detection
od = AdvancedObjectDetection()

cap = cv2.VideoCapture("../videos/Tennis.mp4")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

try:
    while True:
        ret, frame = cap.read()
        print("Frame read")
        if not ret or exit_flag:  # Exit loop if video ends or ESC is pressed
            break
        
        frame = cv2.resize(frame, (192 * 3, 144 * 3))

        # Detect objects on frame
        
        boxes = od.detect(frame)


        # Check if 'f' key was pressed
        if f_pressed:
            f_pressed = False  # Reset flag
            object_selected = False
            print(mouse_x, mouse_y)

            # Ensure the latest mouse position is updated immediately
            cv2.waitKey(1)

            for class_id, box in zip(class_ids, boxes):
                box_area = box[2] * box[3]
                image_area = frame.shape[0] * frame.shape[1]
                if box_area > image_area / 2:
                    continue
                cx, cy = calculate_center(box)
                if box[0] <= mouse_x <= box[0] + box[2] and box[1] <= mouse_y <= box[1] + box[3]:
                    selected_object = box
                    selected_center = (cx, cy)
                    selected_class = od.classes[class_id]  # Get the class name
                    trajectory.clear()  # Clear trajectory when selecting a new object
                    object_selected = True
                    break

            if not object_selected:
                # If no object is selected, reset to show all boxes
                selected_object = None
                selected_center = None
                selected_class = None
                trajectory.clear()  # Clear trajectory when deselecting an object

        # Check if 's' key was pressed
        if s_pressed:
            s_pressed = False  # Reset flag
            if selected_object is not None:  # Only toggle if an object is being followed
                show_trajectory = not show_trajectory
                if not show_trajectory:
                    trajectory.clear()  # Clear trajectory when hiding it

        # If an object is selected, track the nearest box
        if selected_object is not None:
            min_distance = float("inf")
            nearest_box = None
            nearest_center = None
            nearest_class = None

            for class_id, box in zip(class_ids, boxes):
                cx, cy = calculate_center(box)
                distance = euclidean_distance(selected_center, (cx, cy))
                if distance < min_distance:
                    min_distance = distance
                    nearest_box = box
                    nearest_center = (cx, cy)
                    nearest_class = od.classes[class_id]

            if nearest_box is not None:
                selected_object = nearest_box
                selected_center = nearest_center
                selected_class = nearest_class

                # Update trajectory
                if show_trajectory:
                    trajectory.append(selected_center)

            # Draw the selected box with its class
            x, y, w, h = selected_object
            color = (0, 255, 0)  # Green for the selected box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"Selected {selected_class}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Draw all boxes if no object is selected
            for class_id, box in zip(class_ids, boxes):
                x, y, w, h = box
                color = (0, 0, 255)  # Red for all boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                class_name = od.classes[class_id]
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2,)

        # Draw the trajectory
        if show_trajectory and trajectory:
            for point in trajectory:
                cv2.circle(frame, point, 3, (255, 0, 0), -1)  # Blue dots for trajectory

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

except Exception as e:
    print("Exception in main loop:", e)

finally:
    print("\n\nExiting...\n")
    cap.release()
    cv2.destroyAllWindows()

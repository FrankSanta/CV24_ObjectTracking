import cv2
import math
import numpy as np
import threading
from pynput import keyboard

class ObjectDetection:
    def __init__(
        self,
        weights_path="../dnn_model/yolov4.weights",
        cfg_path="../dnn_model/yolov4.cfg",
    ):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 512

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA if available
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception as e:
            print("[WARNING] CUDA is not available. Falling back to CPU.")
            print(e)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(
            size=(self.image_size, self.image_size), scale=1 / 255
        )

    def load_class_names(self, classes_path="../dnn_model/classes.txt"):
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        return self.classes

    def detect(self, frame):
        return self.model.detect(
            frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold
        )

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

def filter_lines(lines):
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate angle of the line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Classify lines based on angle
            if abs(angle) <= 5:  # Horizontal lines (adjust threshold as needed)
                horizontal_lines.append(line)
            elif abs(angle) >= 65:  # Vertical lines
                vertical_lines.append(line)
    return horizontal_lines, vertical_lines

def get_line_params(line):
    x1, y1, x2, y2 = line[0]
    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1
    return A, B, -C

def compute_intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = -Dx / D
        y = -Dy / D
        return [x, y]
    else:
        return None  # Lines are parallel
    

def select_corners(corner_points):

    corner_points = np.array(corner_points)
    print(corner_points)

    # Point with the lowest y and x
    min_y_points = corner_points[corner_points[:, 1] == corner_points[:, 1].min()]
    lowest_y_x_point = min_y_points[np.argmin(min_y_points[:, 0])]
    print(lowest_y_x_point)

    # Point with the highest x among those with the lowest y
    highest_x_lowest_y_point = min_y_points[np.argmax(min_y_points[:, 0])]
    print(highest_x_lowest_y_point)

    # Point with the highest y and x
    max_y_points = corner_points[corner_points[:, 1] == corner_points[:, 1].max()]
    highest_y_x_point = max_y_points[np.argmax(max_y_points[:, 0])]
    print(highest_y_x_point)

    # Point with the lowest x among those with the highest y
    lowest_x_highest_y_point = max_y_points[np.argmin(max_y_points[:, 0])]
    print(lowest_x_highest_y_point)

    selected_points = np.array([
        lowest_y_x_point,
        highest_x_lowest_y_point,
        highest_y_x_point,
        lowest_x_highest_y_point
    ])

    return selected_points

# Start the listener
key_thread = threading.Thread(target=start_listener, daemon=True)
key_thread.start()

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("../videos/Tennis.mp4")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

try:
    while True:
        ret, frame = cap.read()
        if not ret or exit_flag:  # Exit loop if video ends or ESC is pressed
            break
        frame = cv2.resize(frame, (192 * 3, 144 * 3))

        # Detect objects on frame
        (class_ids, scores, boxes) = od.detect(frame)

        # Check if 'f' key was pressed
        if f_pressed:
            f_pressed = False  # Reset flag
            object_selected = False
            print(mouse_x, mouse_y)

            # Ensure the latest mouse position is updated immediately
            cv2.waitKey(1)

            for class_id, box in zip(class_ids, boxes):
                cx, cy = calculate_center(box)
                if (
                    box[0] <= mouse_x <= box[0] + box[2]
                    and box[1] <= mouse_y <= box[1] + box[3]
                ):
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
            if (
                selected_object is not None
            ):  # Only toggle if an object is being followed
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
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        else:
            # Draw all boxes if no object is selected
            for class_id, box in zip(class_ids, boxes):
                x, y, w, h = box
                color = (0, 0, 255)  # Red for all boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                class_name = od.classes[class_id]
                cv2.putText(
                    frame,
                    class_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Draw the trajectory
        if show_trajectory and trajectory:
            for point in trajectory:
                cv2.circle(frame, point, 3, (255, 0, 0), -1)  # Blue dots for trajectory

        if cv2.waitKey(1) == 27:  # ESC key for fallback exit
            exit_flag = True
            break

except Exception as e:
    print("Exception in main loop:", e)

finally:
    print("\n\nExiting...\n")
    cap.release()
    cv2.destroyAllWindows()

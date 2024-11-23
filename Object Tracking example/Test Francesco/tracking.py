import cv2
import math
import numpy as np
import threading
import random
from pynput import keyboard
from sklearn.cluster import KMeans
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, model_path="../dnn_model/yolov8n.pt"):
        print("Loading Object Detection")
        print("Running ultralytics YOLOv8-nano")

        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.classes = self.model.names  # Class names from the model
        self.classes[32] = "tennis ball"
        self.image_size = 512
        self.confThreshold = 0.3
        self.iouThreshold = 0.3

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect(self, frame):
        # Run YOLOv8 inference
        results = self.model(frame, imgsz=self.image_size, conf=self.confThreshold, iou=self.iouThreshold)
        
        # Extract detection results
        detections = results[0]  # YOLOv8 returns a list, take the first (single frame detection)
        boxes = detections.boxes.xyxy.cpu().numpy()  # Convert to NumPy array: [x_min, y_min, x_max, y_max]
        confidences = detections.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = detections.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        
        # Convert boxes to (x, y, w, h) format
        converted_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            x = int(x_min)
            y = int(y_min)
            w = int(x_max - x_min)
            h = int(y_max - y_min)
            converted_boxes.append([x, y, w, h])

        return class_ids, confidences, converted_boxes

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


def mouse_callback(event, x, y, flags, param):
    """
    Tracks the mouse coordinates.
    """
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def calculate_feet_position(box):
    """Calculate the center of a bounding box."""
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h
    return cx, cy


def euclidean_distance(pt1, pt2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


# Function to start the listener in a separate thread
def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def find_ball(frame):
    y1, y2, x1, x2 = 50, 315, 110, 430
    roi = frame[y1:y2, x1:x2]
    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Define the RGB bounds for the pixel values
    lower_bound = np.array([174, 194, 154], dtype=np.uint8)
    upper_bound = np.array([206, 226, 186], dtype=np.uint8)

    # Create a mask for pixels within the defined range
    mask = cv2.inRange(roi, lower_bound, upper_bound)

    # Find the locations of the matching pixels
    matching_pixels = np.column_stack(np.where(mask > 0))

    # Draw yellow circles around the matching pixels
    for (y, x) in matching_pixels:
        cv2.circle(frame, (x1 + x, y1 + y), 5, (0, 255, 255), -1)

    return frame

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
    

#def select_corners(corner_points):

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

def select_corners(corner_points):

    corner_points = np.array(corner_points)
    
    points = []

    for point in corner_points:
        if (point[1] > 50 and point[1] < 70) or (point[1] > 360 and point[1] < 375):
            if (point[0] > 200 and point[0] < 220) or (point[0] > 350 and point[0] < 370) or (point[0] > 490 and point[0] < 510) or (point[0] > 75 and point[0] < 100):
                points.append(point)

    points = np.array(points)
    sorted_points = points[points[:, 0].argsort()] 
    sorted_points = sorted_points[sorted_points[:, 1].argsort(kind='mergesort')]

    selected_points = np.array([
    [sorted_points[0][0], sorted_points[0][1]],
    [sorted_points[1][0], sorted_points[1][1]],
    [sorted_points[3][0], sorted_points[3][1]],
    [sorted_points[2][0], sorted_points[2][1]]], dtype="float32")
    


    return selected_points


# Function to generate a random color (RGB tuple)
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Function to create a new list with 'n' random colors
def create_color_list(n):
    new_colors = [generate_random_color() for _ in range(n)]  # Generate 'n' random colors
    return new_colors

# Start the listener
key_thread = threading.Thread(target=start_listener, daemon=True)
key_thread.start()

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("../videos/Tennis.mp4")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

ball = [] 
sinner = []  
djokovic = []  
y_sinner1 = 200
y_sinner2 = 400
x_sinner1 = 30
x_sinner2 = 530
y_djokovic1 = 80
y_djokovic2 = 170
x_djokovic1 = 160
x_djokovic2 = 400


i = 0
while True:

    ret, frame = cap.read()
    if not ret or exit_flag:  # Exit loop if video ends or ESC is pressed
        break
    frame = cv2.resize(frame, (192 * 3, 144 * 3))

    if i == 0:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Edge detection
        edges = cv2.Canny(blur, 50, 100)
            
            
        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=100, maxLineGap=10)
        line_image = frame.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        horizontal_lines, vertical_lines = filter_lines(lines)
        # Visualize filtered lines
        '''filtered_line_image = frame.copy()
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(filtered_line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for horizontal lines
        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(filtered_line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for vertical lines
        cv2.imshow('Lines Court', filtered_line_image)
        cv2.waitKey(1)'''


        # Compute intersections between horizontal and vertical lines
        intersections = []
        for h_line in horizontal_lines:
            L1 = get_line_params(h_line)
            for v_line in vertical_lines:
                L2 = get_line_params(v_line)
                point = compute_intersection(L1, L2)
                if point is not None and point[0] >= 50 and point[1] >= 50 and point[0] <= frame.shape[1] - 50 and point[1] <= frame.shape[0] - 50:
                    intersections.append(point)

        # Visualize intersections
        '''intersection_image = frame.copy()
        for point in intersections:
            x, y = map(int, point)
            cv2.circle(intersection_image, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow('Intersections', intersection_image)
        cv2.waitKey(1) '''

        # Cluster intersections to find four corners
        points = np.array(intersections)
        n_clusters = 39

        if len(points) >= n_clusters:
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
            corner_points = kmeans.cluster_centers_

            # Visualize clustered corners
            '''cluster_image = frame.copy()
            colors = create_color_list(n_clusters)
            labels = kmeans.labels_
            for idx, point in enumerate(points):
                x, y = map(int, point)
                cluster_idx = labels[idx]
                cv2.circle(cluster_image, (x, y), 5, colors[cluster_idx], -1)
            for idx, center in enumerate(corner_points):
                x, y = map(int, center)
                cv2.circle(cluster_image, (x, y), 10, colors[idx], 2)
            cv2.imshow('Clustered Corners', cluster_image)
            cv2.waitKey(1)'''

            # Order corner points
            src_points = select_corners(corner_points)

            # Compute homography and warp perspective
            width, height = 180, 500
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")

            homography_matrix, status = cv2.findHomography(src_points, dst_points)
            rectified_frame = cv2.warpPerspective(frame, homography_matrix, (width, height))

            cv2.imshow('Rectified Court', rectified_frame)
            cv2.waitKey(1)
            
        else:
            print("Not enough intersection points detected to compute homography.")



    i += 1

    frame = find_ball(frame)

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)

    min_y_diff_sinner = float("inf")
    min_y_diff_djokovic = float("inf")
    sinner_center = None
    djokovic_center = None


    for class_id, box in zip(class_ids, boxes):
        cx, cy = calculate_feet_position(box)
        # Check for "tennis ball" class and save its center
        if od.classes[class_id] == "tennis ball":
            ball.append((cx, cy))

        else:

            if  cy > y_djokovic1 and cy < y_djokovic2 and cx < x_djokovic2 and cx > x_djokovic1:
                djokovic_center = (cx, cy)
            elif cy > y_sinner1 and cy < y_sinner2 and cx < x_sinner2 and cx > x_sinner1:
                sinner_center = (cx, cy)



    # Update the lists with the calculated centers
    if sinner_center:
        sinner.append(sinner_center)
    if djokovic_center:
        djokovic.append(djokovic_center)

    # Draw the trajectory points in blue
    for point in ball:
        cv2.circle(frame, point, 3, (255, 0, 0), -1)  
    for point in sinner:
        cv2.circle(frame, point, 3, (255, 0, 0), -1)  
    for point in djokovic:
        cv2.circle(frame, point, 3, (255, 0, 0), -1)

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
            cx, cy = calculate_feet_position(box)
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
            cx, cy = calculate_feet_position(box)
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
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw the trajectory
    if show_trajectory and trajectory:
        for point in trajectory:
            cv2.circle(frame, point, 3, (255, 0, 0), -1)  # Blue dots for trajectory

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    if ball:
        ball_array = np.array(ball, dtype='float32').reshape(-1, 1, 2)
        transformed_ball = cv2.perspectiveTransform(ball_array, homography_matrix)
        transformed_ball_list = transformed_ball.reshape(-1, 2)
        for point in transformed_ball_list:
            x, y = int(point[0]), int(point[1])
            cv2.circle(rectified_frame, (x, y), 5, (0, 0, 255), -1)

    if sinner:
        sinner_array = np.array(sinner, dtype='float32').reshape(-1, 1, 2)
        transformed_sinner = cv2.perspectiveTransform(sinner_array, homography_matrix)
        transformed_sinner_list = transformed_sinner.reshape(-1, 2)
        for point in transformed_sinner_list:
            x, y = int(point[0]), int(point[1])
            cv2.circle(rectified_frame, (x, y), 5, (0, 255, 0), -1)

    if djokovic:
        djokovic_array = np.array(djokovic, dtype='float32').reshape(-1, 1, 2)
        transformed_djokovic = cv2.perspectiveTransform(djokovic_array, homography_matrix)
        transformed_djokovic_list = transformed_djokovic.reshape(-1, 2)
        for point in transformed_djokovic_list:
            x, y = int(point[0]), int(point[1])
            cv2.circle(rectified_frame, (x, y), 5, (255, 0, 0), -1)

    cv2.imshow('Rectified Court', rectified_frame)
    cv2.waitKey(1)


height, width = rectified_frame.shape[:2]
# Initialize a heatmap array with zeros
heatmap_djokovic = np.zeros((height, width), dtype=np.float32)
heatmap_sinner = np.zeros((height, width), dtype=np.float32)

for x, y in transformed_djokovic_list:
    x = int(round(x))
    y = int(round(y))
    if 0 <= x < width and 0 <= y < height:
        heatmap_djokovic[y, x] += 1  # Increment the count at the position

for x, y in transformed_sinner_list:
    x = int(round(x))
    y = int(round(y))
    if 0 <= x < width and 0 <= y < height:
        heatmap_sinner[y, x] += 1  # Increment the count at the position

# Apply Gaussian blur to smooth the heatmap
heatmap_djokovic_blurred = cv2.GaussianBlur(heatmap_djokovic, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_REPLICATE)
heatmap_sinner_blurred = cv2.GaussianBlur(heatmap_sinner, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_REPLICATE)

heatmap_djokovic_normalized = cv2.normalize(heatmap_djokovic_blurred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
heatmap_sinner_normalized = cv2.normalize(heatmap_sinner_blurred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

heatmap_djokovic_uint8 = heatmap_djokovic_normalized.astype(np.uint8)
heatmap_sinner_uint8 = heatmap_sinner_normalized.astype(np.uint8)

heatmap_djokovic_colored = cv2.applyColorMap(heatmap_djokovic_uint8, cv2.COLORMAP_JET)
heatmap_sinner_colored = cv2.applyColorMap(heatmap_sinner_uint8, cv2.COLORMAP_JET)

cv2.imshow('Heatmap Djokovic', heatmap_djokovic_colored)
cv2.waitKey(0)
cv2.imshow('Heatmap Sinner', heatmap_sinner_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()
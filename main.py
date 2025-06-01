import cv2
import numpy as np

# --- Configuration ---

# Height and width of the entire field
TARGET_WIDTH = 400
TARGET_HEIGHT = 400

# Value of the minimum height and minimum width of the detected region
MIN_BOUNDING_BOX_DIM = 30

# Minimum number of pixels in the region
MIN_CONTOUR_AREA = 800


# HSV color ranges for red
LOWER_RED1 = np.array([0, 130, 100])    # H_min, S_min, V_min
UPPER_RED1 = np.array([10, 255, 255])   # H_max, S_max, V_max
LOWER_RED2 = np.array([160, 130, 100])  # H_min, S_min, V_min
UPPER_RED2 = np.array([180, 255, 255])  # H_max, S_max, V_max

MORPH_KERNEL_SIZE = (7, 7) # Kernel for morphological operations
kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)

# --- Use Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not use Webcam")
    exit()

print("Webcam Launched. Press 'q' to Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read Frame (End of Stream?)")
        break

    # --- Step 1: Crop and scale image to 1:1 aspect ratio ---
    h_orig, w_orig = frame.shape[:2]
    if w_orig > h_orig:
        diff = (w_orig - h_orig) // 2
        cropped_frame = frame[:, diff:diff + h_orig]
    elif h_orig > w_orig:
        diff = (h_orig - w_orig) // 2
        cropped_frame = frame[diff:diff + w_orig, :]
    else:
        cropped_frame = frame
    display_image = cv2.resize(cropped_frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # --- Step 2: Isolate red pixels ---
    hsv_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv_image, LOWER_RED1, UPPER_RED1)
    mask_red2 = cv2.inRange(hsv_image, LOWER_RED2, UPPER_RED2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # --- Step 3: Close gaps & remove shit ---
    closed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Step 4: Blob detection and filtering by size ---
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_visualization_frame = display_image.copy()
    found_objects_this_frame = False

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Filter 1: Minimum area of the contour
        if area > MIN_CONTOUR_AREA:
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)

            # Filter 2: Minimum dimensions of the bounding box
            if w_rect >= MIN_BOUNDING_BOX_DIM and h_rect >= MIN_BOUNDING_BOX_DIM:
                if not found_objects_this_frame:
                    print("\n--- Detected red objects ---")
                    found_objects_this_frame = True

                (x_circle, y_circle), radius_circle = cv2.minEnclosingCircle(contour)
                center_circle = (int(x_circle), int(y_circle))
                radius_circle = int(radius_circle)

                print(f"Object ID {i}: Center=({center_circle[0]}, {center_circle[1]}), Radius={radius_circle} (BBox: {w_rect}x{h_rect})")

                # --- Step 5: Circle red areas in blue ---
                cv2.circle(output_visualization_frame, center_circle, radius_circle, (255, 0, 0), 2)
                cv2.putText(output_visualization_frame, str(i), (center_circle[0] - 10, center_circle[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Show live video window ---
    cv2.imshow('1: original', display_image)
    cv2.imshow('4: Detected objects', output_visualization_frame)
    cv2.imshow('2: Isolate red pixels', red_mask)
    cv2.imshow('3: further processing', processed_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- close Program ---
cap.release()
cv2.destroyAllWindows()
print("Exit")
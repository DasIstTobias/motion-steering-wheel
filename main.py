# Use this script with:
# ```
# sudo WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
#      XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
#      QT_QPA_PLATFORM=wayland \
#      python3.11 script_name.py
# ```
# Otherwise it will not work.

import cv2
import numpy as np
import time
from evdev import UInput, AbsInfo
from evdev import ecodes as e

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

VENDOR_ID = 0x045e
PRODUCT_ID = 0x02dd
DEVICE_NAME = "Xbox Wireless Controller"

OPENCV_MIN_X = 0.0
OPENCV_MAX_X = float(TARGET_WIDTH)

# Joystick axis values
JOYSTICK_MIN = -32768
JOYSTICK_MAX = 32767
JOYSTICK_CENTER = 0

def map_opencv_to_joystick(opencv_pos: float, min_opencv_val: float, max_opencv_val: float) -> int:

    # Ensure that the value is within the defined range
    clamped_pos = max(min_opencv_val, min(opencv_pos, max_opencv_val))
    
    # (max_val - clamped_pos) reverses direction
    normalized_pos = (max_opencv_val - clamped_pos) / (max_opencv_val - min_opencv_val)

    # Scale to the joystick area
    joystick_value = JOYSTICK_MIN + normalized_pos * (JOYSTICK_MAX - JOYSTICK_MIN)
    
    return int(round(joystick_value))

if __name__ == "__main__":
    ui = None
    cap = None

    try:
        # --- Setup Virtual Controller ---
        _DEVICE_NAME = "Xbox Wireless Controller"
        _VENDOR_ID = 0x045e
        _PRODUCT_ID = 0x02dd
        _VERSION = 0x1
        
        print("DEBUG: Defining full controller capabilities using AbsInfo")
        capabilities = {
            e.EV_KEY: [
                e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y,
                e.BTN_TL, e.BTN_TR, e.BTN_TL2, e.BTN_TR2,
                e.BTN_SELECT, e.BTN_START, e.BTN_MODE,
                e.BTN_THUMBL, e.BTN_THUMBR
            ],
            e.EV_ABS: [
                (e.ABS_X, AbsInfo(value=0, min=-32767, max=32767, fuzz=16, flat=32, resolution=0)), # value=JOYSTICK_CENTER
                (e.ABS_Y, AbsInfo(value=0, min=-32767, max=32767, fuzz=16, flat=32, resolution=0)), # Left Stick Y
                
                (e.ABS_RX, AbsInfo(value=0, min=-32767, max=32767, fuzz=16, flat=32, resolution=0)),# Right Stick X
                (e.ABS_RY, AbsInfo(value=0, min=-32767, max=32767, fuzz=16, flat=32, resolution=0)),# Right Stick Y
                
                (e.ABS_Z, AbsInfo(value=0, min=0, max=1023, fuzz=0, flat=0, resolution=0)),  # Left Trigger (LT)
                (e.ABS_RZ, AbsInfo(value=0, min=0, max=1023, fuzz=0, flat=0, resolution=0)), # Right Trigger (RT)
                
                (e.ABS_HAT0X, AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)), # D-Pad X
                (e.ABS_HAT0Y, AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0))  # D-Pad Y
            ],
            e.EV_MSC: [e.MSC_SCAN],
        }
        
        print(f"Try to create a virtual controller: {_DEVICE_NAME}...")
        
        print("This script requires root privileges")
        
        ui = UInput(capabilities, name=DEVICE_NAME, vendor=VENDOR_ID, product=PRODUCT_ID, version=0x1)
        print("Virtual controller created. Starting Webcam...")

        # --- Use Webcam ---
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not use Webcam")
            exit()

        print("Webcam Launched. Press 'q' to Exit Program.")
        
        current_opencv_x_pos = OPENCV_MAX_X / 2.0 

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
            
            found_object_for_steering = False # Flag indicating whether an object for the control was found in this frame

            object_centers_x = [] # Collects X-coordinates for debugging or extended logic

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                # Filter 1: Minimum area of the contour
                if area > MIN_CONTOUR_AREA:
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)

                    # Filter 2: Minimum dimensions of the bounding box
                    if w_rect >= MIN_BOUNDING_BOX_DIM and h_rect >= MIN_BOUNDING_BOX_DIM:
                        (x_circle, y_circle), radius_circle = cv2.minEnclosingCircle(contour)
                        center_circle = (int(x_circle), int(y_circle))
                        radius_circle = int(radius_circle)
                        
                        object_centers_x.append(center_circle[0])

                        # Using the FIRST valid object found for the controller
                        if not found_object_for_steering:
                            current_opencv_x_pos = float(center_circle[0])
                            found_object_for_steering = True
                            # Mark the controlling object in special way
                            cv2.circle(output_visualization_frame, center_circle, radius_circle + 5, (0, 255, 0), 2) # Grüner Kreis um steuerndes Objekt

                        # --- Step 5: Circle red areas in blue ---
                        cv2.circle(output_visualization_frame, center_circle, radius_circle, (255, 0, 0), 2)
                        cv2.putText(output_visualization_frame, str(i), (center_circle[0] - 10, center_circle[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            if found_object_for_steering:
                 pass
            elif not object_centers_x:
                current_opencv_x_pos = OPENCV_MAX_X / 2.0

            # --- Joystick emulation update ---
            joystick_x_axis_value = map_opencv_to_joystick(current_opencv_x_pos, OPENCV_MIN_X, OPENCV_MAX_X)
            
            # Send joystick data
            ui.write(e.EV_ABS, e.ABS_X, joystick_x_axis_value)
            # Keep other axes centred to avoid unwanted movements
            ui.write(e.EV_ABS, e.ABS_Y, JOYSTICK_CENTER) # Left stick Y
            ui.write(e.EV_ABS, e.ABS_RX, JOYSTICK_CENTER) # Right stick X
            ui.write(e.EV_ABS, e.ABS_RY, JOYSTICK_CENTER) # Right stick Y
            ui.syn() # Synchronise events

            print(f"OpenCV X: {current_opencv_x_pos:6.1f} (Range {OPENCV_MIN_X}-{OPENCV_MAX_X}) -> Joystick X: {joystick_x_axis_value:6} (Range {JOYSTICK_MIN}-{JOYSTICK_MAX})  ", end='\r')

            # --- Show live video window ---
            # cv2.imshow('1: original', display_image)
            cv2.imshow('Detected objects', output_visualization_frame)
            # cv2.imshow('2: Isolate red pixels', red_mask)
            # cv2.imshow('3: further processing', processed_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n'q' pressed, exiting...")
                break
            
            time.sleep(0.01)

    except PermissionError:
        print("\nERROR: No authorisation to access /dev/uinput.")
    except FileNotFoundError:
        print("\nERROR: /dev/uinput not found. Make sure that the ‘uinput’ kernel module is loaded.")
        print("Try: 'sudo modprobe uinput'")
    except Exception as ex:
        print(f"\nUff, Something bad has happened: {ex}")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        if ui is not None:
            ui.close()
        print("Exit")

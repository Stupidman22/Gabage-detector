# Import necessary modules
import os
import time
import cv2
import torch
import onnxruntime as rt
from PIL import Image
from torchvision import transforms
import serial
from datetime import datetime

# Constants
MODEL_PATH = "models/gc_torchscript.onnx"
CLASS_NAMES = ["cardboard_paper", "glass", "metal", "others", "plastic"]
DEVICE = "cpu"
ARDUINO_PORT = "/dev/ttyUSB0"  # or "/dev/ttyUSB0" for Linux/RPi5
BAUD_RATE = 9600
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
IMAGE_SIZE = (394, 394)
ZOOM_SCALE = 1.01  # Zoom factor, adjust to zoom in more or less
OUTPUT_PATH = "./output"

# Initialize ONNX session
sess_opt = rt.SessionOptions()
sess_opt.intra_op_num_threads = 4
ort_session = rt.InferenceSession(MODEL_PATH, sess_opt, providers=["CPUExecutionProvider"])

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classifier with serial communication
class GarbageClassifier:
    def __init__(self):
        try:
            self.ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        except serial.SerialException as e:
            print(f"Warning: Could not open serial port {ARDUINO_PORT}. Error: {e}")
            self.ser = None
        self.last_sent_time = 0  # To track cooldown period

    def send_to_arduino(self, class_label):
        class_to_int = {
            "cardboard_paper": "1",
            "glass": "2",
            "metal": "3",
            "others": "4",
            "plastic": "5"
        }
        class_int = class_to_int.get(class_label, "0")
        cooldown_time = 1  # Cooldown period in seconds
        # Check cooldown period
        if time.time() - self.last_sent_time < cooldown_time:
            print("Cooldown active. Skipping this detection.")
            return

        if self.ser and self.ser.is_open:
            self.ser.write(class_int.encode())
            print(f"Sent {class_int} to Arduino for class '{class_label}'")
            self.last_sent_time = time.time()

    def close_serial(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def classify_image(self, img):
        img = data_transforms(img).unsqueeze(0).to(DEVICE)
        ort_inputs = {ort_session.get_inputs()[0].name: img.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = torch.tensor(ort_outs[0])
        _, preds = torch.max(outputs, 1)
        return CLASS_NAMES[preds[0]]


def zoom_and_crop(frame, zoom_scale=ZOOM_SCALE, target_size=IMAGE_SIZE):
    # Get the center of the image
    center_x, center_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2

    # Calculate the zoomed size
    zoomed_width = int(FRAME_WIDTH * zoom_scale)
    zoomed_height = int(FRAME_HEIGHT * zoom_scale)

    # Resize the image to zoom in
    zoomed_frame = cv2.resize(frame, (zoomed_width, zoomed_height))

    # Calculate the crop area to get the center region
    crop_x1 = max(center_x - target_size[0] // 2, 0)
    crop_y1 = max(center_y - target_size[1] // 2, 0)
    crop_x2 = min(crop_x1 + target_size[0], zoomed_width)
    crop_y2 = min(crop_y1 + target_size[1], zoomed_height)

    # Crop the zoomed image to the target size
    cropped_frame = zoomed_frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Ensure the cropped frame is the correct size
    cropped_frame = cv2.resize(cropped_frame, target_size)

    return cropped_frame
def process_frame(frame, classifier):
    # Add a short delay to allow the camera to focus
    time.sleep(0.5)  # Adjust the time (in seconds) as needed

    # Crop the frame to the desired region
    # cropped_frame = frame[:, (FRAME_WIDTH - FRAME_HEIGHT) // 2:(FRAME_WIDTH - FRAME_HEIGHT) // 2 + FRAME_HEIGHT]
    cropped_frame = zoom_and_crop(frame)
    img_path = "capture.jpg"
    cv2.imwrite(img_path, cropped_frame)
    img = Image.open(img_path)

    # Classify and send result
    start_time = time.time()
    pred_class = classifier.classify_image(img)
    classifier.send_to_arduino(pred_class)
    print(f"Predicted class: {pred_class} | Inference time: {time.time() - start_time:.4f} seconds")

    # Display prediction on frame
    cv2.putText(cropped_frame, pred_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Captured Image", cropped_frame)

    # Save image
    output_path = os.path.join(OUTPUT_PATH, pred_class, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    img.save(output_path)

def main():
    classifier = GarbageClassifier()
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for name in CLASS_NAMES:
        os.makedirs(os.path.join(OUTPUT_PATH, name), exist_ok=True)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    # Check if the camera supports autofocus (CAP_PROP_AUTOFOCUS)
    autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) is not None
    print(f"Autofocus Supported: {autofocus_supported}")

    # Try enabling autofocus if supported
    if autofocus_supported:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 1 to enable autofocus
    else:
        print("Warning: Autofocus is not supported by this camera.")

    print("1. Press the space key to capture an image or wait for a signal from the Arduino.")
    print("2. Press the Q key to exit the program.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            feed_frame = frame.copy()
            cv2.rectangle(feed_frame, ((FRAME_WIDTH-FRAME_HEIGHT)//2, 0),
                          ((FRAME_WIDTH-FRAME_HEIGHT)//2 + FRAME_HEIGHT, FRAME_HEIGHT), (0, 255, 0), 1)
            cv2.imshow("Video Feed", feed_frame)

            key = cv2.waitKey(1) & 0xFF
            if classifier.ser and classifier.ser.in_waiting > 0:
                data = classifier.ser.readline().decode('utf-8').strip()
                print(f"Received from Arduino: {data}")
                if data == "0":
                    process_frame(frame, classifier)

            if key == ord(' '):
                process_frame(frame, classifier)

            if key == ord('q'):
                print("Exiting...")
                break

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        classifier.close_serial()

if __name__ == "__main__":
    main()

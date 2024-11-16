import os
import time
import cv2
import torch
import onnxruntime as rt
from PIL import Image
from torchvision import transforms
import serial
import hyperparams as hparams

# Constants
MODEL_PATH = "models/gc_torchscript.onnx"
CLASS_NAMES = ["cardboard_paper", "glass", "metal", "others", "plastic"]
DEVICE = "cpu"

# Serial communication setup (Update port for your system)
ARDUINO_PORT = "COM3"  # or "/dev/ttyUSB0" for Linux/RPi5
BAUD_RATE = 9600

# Model and camera settings
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CAPTURE_FPS = 30
IMAGE_SIZE = (394, 394)  # Expected model input size

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

# Classifier class with serial communication
class GarbageClassifier:
    def __init__(self):
        # Initialize serial connection if available
        self.ser = None
        try:
            self.ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        except serial.SerialException as e:
            print(f"Warning: Could not open serial port {ARDUINO_PORT}. Error: {e}")

    def send_to_arduino(self, class_label):
        """Maps class label to integer and sends it to Arduino via serial."""
        class_to_int = {
            "cardboard_paper": "1",
            "glass": "2",
            "metal": "3",
            "others": "4",
            "plastic": "5"
        }
        class_int = class_to_int.get(class_label, "0")  # Default to "0" if not found

        if self.ser and self.ser.is_open:
            self.ser.write(class_int.encode())
            print(f"Sent {class_int} to Arduino for class '{class_label}'")

    def close_serial(self):
        """Close the serial connection if open."""
        if self.ser and self.ser.is_open:
            self.ser.close()

    def classify_image(self, img):
        """Processes and classifies an image, returning the class label."""
        img = data_transforms(img).unsqueeze(0).to(DEVICE)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = torch.tensor(ort_outs[0])
        _, preds = torch.max(outputs, 1)
        return CLASS_NAMES[preds[0]]

# Video capture and main loop
def main():
    classifier = GarbageClassifier()
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Press 'space' to capture an image, 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Display capture with rectangle
            feed_frame = frame.copy()
            cv2.rectangle(feed_frame, ((FRAME_WIDTH - FRAME_HEIGHT) // 2, 0),
                          ((FRAME_WIDTH - FRAME_HEIGHT) // 2 + FRAME_HEIGHT, FRAME_HEIGHT), (0, 255, 0), 1)
            cv2.imshow("Video Feed", feed_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space key to capture and classify
                # Crop and save the frame for classification
                cropped_frame = frame[:, (FRAME_WIDTH - FRAME_HEIGHT) // 2:(FRAME_WIDTH - FRAME_HEIGHT) // 2 + FRAME_HEIGHT]
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

            elif key == ord('q'):  # 'q' to quit
                break
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        classifier.close_serial()

if __name__ == "__main__":
    main()

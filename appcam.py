import cv2
import argparse
import numpy as np
from ultralytics import YOLO

# Define zone polygons with a margin
margin = 0.25
ZONE_POLYGON = np.array([
    [0, 0],
    [0.5 - margin, 0],
    [0.5 - margin, 1],
    [0, 1]
])

ZONE_POLYGON_ = np.array([
    [0.5 + margin, 0],
    [1, 0],
    [1, 1],
    [0.5 + margin, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int,
        help="Set the webcam resolution, default is 1280x720"
    )
    return parser.parse_args()

def draw_polygon(frame, polygon, color, thickness):
    """Draw a polygon on the frame."""
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)

def annotate_frame(frame, detections, model_names):
    """Annotate the frame with bounding boxes and labels."""
    for *box, confidence, class_id in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model_names[class_id]} {confidence:.2f}"
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put the label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load YOLO model
    model = YOLO("best.pt")

    # Scale polygons to frame size
    zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
    zone_polygon_ = (ZONE_POLYGON_ * np.array([frame_width, frame_height])).astype(int)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Unable to read frame from the webcam.")
                break

            # Run YOLO model and filter results
            results = model(frame, agnostic_nms=True)[0]
            detections = results.boxes.xyxy.cpu().numpy()  # Get bounding boxes
            confidences = results.boxes.conf.cpu().numpy()  # Get confidences
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Get class IDs

            # Combine into a single array
            detections = np.hstack((detections, confidences[:, None], class_ids[:, None]))

            # Draw bounding boxes and labels
            annotate_frame(frame, detections, model.names)

            # Draw polygons
            draw_polygon(frame, zone_polygon, color=(0, 0, 255), thickness=3)
            draw_polygon(frame, zone_polygon_, color=(255, 0, 0), thickness=3)

            # Show the frame
            cv2.imshow("YOLOv8 Live Detection", frame)

            # Exit on 'ESC' key
            if cv2.waitKey(30) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

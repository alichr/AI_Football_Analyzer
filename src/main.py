import torch
import cv2
from data_pipeline.video_pipeline import Preprocessor, VideoDataset
from detection.detector import YoloDetector
from tracking.tracker import DeepSortTracker  # Updated to use DeepSORT


def main():
    # Initialize video path and model parameters
    video_path = "../data/raw/videos/sample.mp4"
    model_path = "../checkpoint/yolo11n.pt"
    img_size = 640

    # Initialize preprocessor and dataset
    preprocessor = Preprocessor(img_size=img_size)
    dataset = VideoDataset(video_path, preprocessor)

    # Initialize YOLO detector and DeepSORT tracker
    detector = YoloDetector(model_path=model_path, img_size=img_size)
    tracker = DeepSortTracker()

    # Process video frames
    for i, processed_frame in enumerate(dataset):
        tensor = torch.from_numpy(processed_frame).unsqueeze(0)  # [1, 3, H, W]
        print(f"Frame {i} - Shape: {tensor.shape}, dtype: {tensor.dtype}")

        # Perform detection
        results = detector.detect(tensor)
        result = results[0]

        # Extract detection boxes: [x1, y1, x2, y2, conf]
        detections = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            detections.append([*xyxy, conf])

        # Convert frame to BGR for DeepSORT
        original_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # Perform tracking
        tracked_objects = tracker.update(original_frame_bgr, detections)

        # Draw tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            cv2.rectangle(original_frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_frame_bgr, f'ID {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the frame
        cv2.imshow("Tracked", original_frame_bgr)
        cv2.waitKey(5)

        if i == 500:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

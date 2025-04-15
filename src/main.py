import torch
import cv2
import numpy as np
import datetime
from data_pipeline.video_preprocessor import Preprocessor, VideoDataset
from tracking.tracker import YoloTracker

# Constants
CONFIDENCE_THRESHOLD = 0.5
RED = (0, 0, 255)

def main():
    # Initialize video path and model parameters
    video_path = "../data/raw/videos/sample.mp4"
    model_path = "../checkpoint/yolo11_football_20250415_1837222/weights/best.pt"
    img_size = 640

    # Initialize preprocessor and dataset
    preprocessor = Preprocessor(img_size=img_size)
    dataset = VideoDataset(video_path, preprocessor)

    # Get video frame rate
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Initialize YOLO detector and player tracker
    model = YoloTracker(model_path=model_path, img_size=img_size)

    # Process video frames
    for i, processed_frame in enumerate(dataset):
        start = datetime.datetime.now()
        
        # Detection
        tensor = torch.from_numpy(processed_frame).unsqueeze(0)  # [1, 3, H, W]
        results = model.track(tensor, conf=CONFIDENCE_THRESHOLD, persist=True, tracker="bytetrack.yaml")

        tracked_frame = results[0].plot()
        
        # Calculate and display FPS
        end = datetime.datetime.now()
        fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(tracked_frame, fps_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
        
        # Show frame
        cv2.imshow('YOLO Tracking with Data Pipeline', tracked_frame)

      
        
        if cv2.waitKey(1) == ord('q'):
            break
            
        if i == 500:  # Limit to 500 frames for testing
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

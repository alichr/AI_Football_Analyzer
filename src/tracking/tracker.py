from ultralytics import YOLO
import torch

class YoloTracker:
    def __init__(self, model_path='yolov8n.pt', device=None, img_size=640):
        self.model = YOLO(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size

    def detect(self, image_tensor):
        """
        image_tensor: torch.Tensor of shape [1, 3, H, W], normalized [0, 1]
        """
        # model.predict handles device internally
        results = self.model.predict(image_tensor, imgsz=self.img_size, verbose=False, device=self.device)
        return results
        
    def track(self, image_tensor, conf=0.5, tracker="bytetrack.yaml", persist=True):
        """
        Track objects in a sequence of frames
        
        Args:
            image_tensor: torch.Tensor of shape [1, 3, H, W], normalized [0, 1]
            conf: Confidence threshold for detection
            tracker: Tracker to use ("bytetrack.yaml", "botsort.yaml", etc.)
            persist: Whether to persist tracks between frames
            
        Returns:
            Results object from YOLO model with tracking information
        """
        results = self.model.track(
            image_tensor, 
            imgsz=self.img_size, 
            conf=conf, 
            device=self.device, 
            tracker=tracker, 
            persist=persist,
            verbose=False
        )
        return results

if __name__ == "__main__":
    # Example usage
    detector = YoloTracker(model_path='checkpoint/yolo11n.pt', img_size=640)
    dummy_tensor = torch.randn(1, 3, 640, 640)  # Dummy tensor for testing
    results = detector.detect(dummy_tensor)
    
    for result in results:
        print(result)
        # Process the detection results as needed
        
    # Tracking example
    tracking_results = detector.track(dummy_tensor, conf=0.5)
    
    for result in tracking_results:
        print(result)
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            print(f"Track IDs: {result.boxes.id.int().cpu().tolist()}")
        # Process the tracking results as needed
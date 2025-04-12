from ultralytics import YOLO
import torch

class YoloDetector:
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

if __name__ == "__main__":
    # Example usage
    detector = YoloDetector(model_path='checkpoint/yolo11n.pt', img_size=640)
    dummy_tensor = torch.randn(1, 3, 640, 640)  # Dummy tensor for testing
    results = detector.detect(dummy_tensor)
    
    for result in results:
        print(result)
        # Process the detection results as needed
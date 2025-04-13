import cv2
import numpy as np
import torch

# -------------------------------
# 1. Frame Loader (VideoReader)
# -------------------------------
class VideoLoader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

# -------------------------------
# 2. Preprocessing for YOLO input
# -------------------------------
class Preprocessor:
    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, frame):
        h, w = frame.shape[:2]
        scale = self.img_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (nw, nh))

        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized

        padded = padded[..., ::-1]  # BGR to RGB
        padded = padded.astype(np.float32) / 255.0
        padded = np.transpose(padded, (2, 0, 1))  # HWC → CHW

        return padded

# -------------------------------
# 3. Dataset-style Video Wrapper
# -------------------------------
class VideoDataset:
    def __init__(self, video_path, preprocessor):
        self.loader = VideoLoader(video_path)
        self.preprocessor = preprocessor

    def __iter__(self):
        for frame in self.loader:
            yield self.preprocessor(frame)

# -------------------------------
# 4. Example Usage
# -------------------------------
if __name__ == "__main__":
    video_path = "data/raw/videos/sample.mp4"
    img_size = 640

    preprocessor = Preprocessor(img_size=img_size)
    dataset = VideoDataset(video_path, preprocessor)

    for i, processed_frame in enumerate(dataset):
        tensor = torch.from_numpy(processed_frame).unsqueeze(0)  # [1, 3, H, W]
        print(f"Frame {i} - Shape: {tensor.shape}, dtype: {tensor.dtype}")

        # visualize the processed frame
        cv2.imshow("Processed Frame", processed_frame.transpose(1, 2, 0)[..., ::-1])
        cv2.waitKey(5)


        # For YOLO inference:
        # results = model(tensor)  # If using Ultralytics or custom YOLO

        if i == 500:
            break  # Just show a few frames

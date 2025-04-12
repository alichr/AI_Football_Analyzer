from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3)

    def update(self, frame, detections):
        """
        frame: Original image (BGR) as numpy array [H, W, 3]
        detections: list of [x1, y1, x2, y2, conf] from YOLO
        returns: list of tracked objects with ID and bbox
        """
        if len(detections) == 0:
            return []

        bbox_xywh = []
        confidences = []

        for det in detections:
            x1, y1, x2, y2, conf = det
            w, h = x2 - x1, y2 - y1
            bbox_xywh.append([x1 + w/2, y1 + h/2, w, h])
            confidences.append(conf)

        tracks = self.tracker.update_tracks(bbox_xywh, confidences, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            results.append([l, t, l + w, t + h, track_id])
        return results

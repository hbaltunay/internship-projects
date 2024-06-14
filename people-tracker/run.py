import numpy as np
from ultralytics import YOLO
from core import process_video, process_frame


if __name__ == "__main__":

    source_video_path = "./test.mp4"
    target_video_path = "./result.mp4"

    # Available model weights: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO(".models/yolov8s.pt")

    class_id = [0]  # Person

    in_area = np.array([[685,775], [1255, 485], [1225, 470], [660, 745]])
    out_area = np.array([[655, 740], [1220, 465], [1195, 445], [635, 715]])

    areas = [in_area, out_area]

    process_video(
        model=model,
        source_path=source_video_path,
        target_path=target_video_path,
        polygons=areas,
        callback=process_frame,
    )

import numpy as np
from ultralytics import YOLO
from core import process_video, process_frame


if __name__ == "__main__":

    source_video_path = "./test.mp4"
    target_video_path = "./result.mp4"

    # Available model weights: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO("C:/Users/HSN/PycharmProjects/AYVOS/models/yolov8x.pt")

    class_id = [0]  # Person

    area = np.array([[2, 1010], [2, 1078], [1380, 1078], [1550, 750], [1850, 720], [1910, 320], [1400, 295],
                     [1090, 450], [935, 475], [770, 450], [385, 590], [425, 798]])

    process_video(
        model=model,
        source_path=source_video_path,
        target_path=target_video_path,
        polygons=area,
        callback=process_frame,
    )
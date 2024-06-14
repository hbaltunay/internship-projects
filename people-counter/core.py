import cv2
import numpy as np


def get_detections(results):
    """
    :param results: It is the value of the results detected by the framework of the given model.
    :return: Returns the results to be used in dictionary form.
    """
    detections = dict()
    detections["xyxy"] = results.boxes.xyxy.cpu().numpy()
    detections["xywh"] = results.boxes.xywh.cpu().numpy()
    detections["confidence"] = results.boxes.conf.cpu().numpy()
    detections["class_id"] = results.boxes.cls.cpu().numpy().astype(int)

    return detections


def get_polygon(frame, points, show):
    """
    :param frame: Video frame to be processed.
    :param points: Coordinates of the area to be detected.
    :param show: Drawing the area on the frame.
    :return: Returns a masked image of the area to be detected.
    """
    if show:
        cv2.polylines(
            frame, [points], isClosed=True, color=(0, 255, 0), thickness=2
        )
    res = np.zeros(frame.shape[:2], np.uint8)
    rect = cv2.boundingRect(points)
    x, y, w, h = rect

    points = points - points.min(axis=0)
    croped = frame[y:y + h, x:x + w].copy()
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res[y:y + h, x:x + w] = mask

    return res


def count_polygon(mask, frame, detections):
    """
    :param mask: Mask image of the area to be detected.
    :param frame: Video frame to be processed.
    :param detections: Dictionary of detection results to be used.
    :return: Returns the procesed frame and detections inside the area.
    """
    count = 0
    in_detect = dict()
    in_detect["xyxy"] = []
    in_detect["xywh"] = []

    for xyxy, xywh in zip(detections["xyxy"], detections["xywh"]):
        x, y, _, _ = xyxy
        if mask[y][x] == 255:
            count += 1
            in_detect["xyxy"].append(xyxy)
            in_detect["xywh"].append(xywh)

    cv2.rectangle(frame, (1520, 900), (1900, 1060), color=(255, 0, 0), thickness=-1)
    cv2.putText(frame, "People Counter", (1525, 950), 5, 2, (255, 255, 255), 2)
    cv2.putText(frame, str(count), (1690, 1025), 2, 2, (255, 255, 255), 2, bottomLeftOrigin=False)

    return frame, in_detect


def show_boxes(frame, in_detect):
    """
    :param frame: Video frame to be processed.
    :param in_detect: Detections in the area.
    :return: Returns the procesed frame.
    """
    for i in range(len(in_detect["xyxy"])):
        x1, y1, _, _ = in_detect["xyxy"][i].astype(int)
        color = (0, 0, 255)

        frame = cv2.ellipse(
            img=frame,
            center=(x1, y1 - 10),
            axes=((in_detect["xywh"][i][-2] / 2).astype(int), (in_detect["xywh"][i][-1] / 8).astype(int)),
            angle=10,
            startAngle=0,
            endAngle=360,
            color=color,
            thickness=2,
        )

    return frame


def to_center_base(detections):
    """
    :param detections: Dictionary of detection results to be used.
    :return: None
    """
    res = []
    for arr in detections["xyxy"]:
        x = int((arr[0] + arr[2]) // 2)
        y = int((arr[1] + arr[3]) // 2)

        c = int((arr[3] - arr[1]) // 2)

        res.append([x, y + c, x, y + c])

    detections["xyxy"] = np.array(res)


def process_frame(frame: np.ndarray, zones: list, results):
    """
    :param frame: Video frame to be processed.
    :param zones: Zones to be estimated in video frames.
    :return: Processed video frame.
    """

    detections = get_detections(results)

    to_center_base(detections)

    frame, in_detect = count_polygon(zones, frame, detections)

    frame = show_boxes(frame, in_detect)

    return frame, detections


def process_video(model, callback, source_path, target_path, polygons) -> None:
    """
    :param model: Model that detects people.
    :param callback: Function to be applied to each frame.
    :param source_path: The path of the source video
    :param target_path: The path of the video to save after processing.
    :param polygons: Coordinates of polygonal areas.
    :return: The final video frames is saved.
    """

    cap = cv2.VideoCapture(source_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    for result in model.predict(source_path, stream=True, imgsz=1280, classes=[0]):  # imgsz=1920

        frame = result.orig_img
        zones = get_polygon(frame, polygons, show=True)
        frame, _ = callback(frame, zones, result)
        out.write(frame)

    cap.release()
    out.release()

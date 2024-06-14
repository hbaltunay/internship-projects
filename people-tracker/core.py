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
    detections["track_id"] = results.boxes.id.int().cpu().numpy()

    return detections


def get_polygon(frame, points, show):
    """
    :param frame: Video frame to be processed.
    :param points: Coordinates of the area to be detected.
    :param show: Drawing the area on the frame.
    :return: Returns a masked image of the area to be detected.
    """
    zones = []
    for pts in points:
        if show:
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=2)

        res = np.zeros(frame.shape[:2], np.uint8)
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect

        pts = pts - pts.min(axis=0)
        croped = frame[y:y+h, x:x+w].copy()
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        res[y:y+h, x:x+w] = mask

        zones.append(res)

    return zones


def count_polygon(mask, frame, detections, polygons, in_ids, out_ids):
    """
    :param mask: Mask image of the area to be detected.
    :param frame: Video frame to be processed.
    :param detections: List of detected values.
    :param polygons: Coordinates of polygonal areas.
    :param in_ids: List of ids.
    :param out_ids: List of ids.
    :return: Processed video frame and id lists.
    """

    for xywh, tid in zip(detections["xywh"], detections["track_id"]):
        x, y, _, _ = list(map(int, xywh))
        if mask[0][y][x] == 255 and not(tid in out_ids) :
            in_ids.append(tid)
            cv2.fillPoly(frame, [polygons[0]], (0,255,0))
        if mask[1][y][x] == 255 and not(tid in in_ids) :
            out_ids.append(tid)
            cv2.fillPoly(frame, [polygons[1]], (0,0,255))

    cv2.rectangle(frame, (550, 825), (700, 875), color=(0, 255, 0), thickness=-1)
    cv2.rectangle(frame, (425, 675), (575, 725), color=(0, 0, 255), thickness=-1)

    cv2.putText(frame, "in: " + str(len(set(in_ids))), (575, 860), 4, 1, (255, 255, 255))
    cv2.putText(frame, "out: " + str(len(set(out_ids))), (440, 710), 4, 1, (255, 255, 255))

    return frame, in_ids, out_ids


def show_boxes(frame, detections):
    """
    :param frame: Video frame to be processed.
    :param detections: Detections in the area.
    :return: Returns the procesed frame.
    """

    for i in range(len(detections["xyxy"])):
        x1, y1, x2, y2 = detections["xyxy"][i].astype(int)
        color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, 1)
        cv2.rectangle(frame, (x1, y1-15), (x1+60, y1), (255, 255, 255), thickness=-1)
        cv2.circle(frame, (detections["xywh"][i].astype(int)[:2]), 2, (0, 0, 255), 2)
        cv2.putText(frame, "ID: " + str(detections["track_id"][i]), (x1+2, y1-2), 1, 1, (0,0,0))

    return frame


def process_frame(frame, zones, results, polygons, in_ids, out_ids):
    """
    :param frame: Video frame to be processed.
    :param zones: Zones to be estimated in video frames.
    :param results: It is the value of the results detected by the framework of the given model.
    :param polygons: Coordinates of polygonal areas.
    :param in_ids: List of ids.
    :param out_ids: List of ids.
    :return: Processed video frame.
    """

    detections = get_detections(results)

    frame, in_ids, out_ids = count_polygon(zones, frame, detections, polygons, in_ids, out_ids)

    frame = show_boxes(frame, detections)

    return frame, in_ids, out_ids


def process_video(model, source_path, target_path, polygons, callback) -> None:
    """
    :param model: The path of the pre-trained model
    :param source_path: The path of the source video
    :param target_path: The path of the video to save after processing.
    :param polygons: Coordinates of polygonal areas.
    :param callback: Function to be applied to each frame.
    :return: The final video frames is saved.
    """

    cap = cv2.VideoCapture(source_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    in_ids = []
    out_ids = []

    for result in model.track(source_path, stream=True, persist=True, imgsz=1440, tracker="bytetrack.yaml", classes=[0]):

        frame = result.orig_img

        zones = get_polygon(frame, polygons, show=True)

        frame, in_ids, out_ids = callback(frame, zones, result, polygons, in_ids, out_ids)

        out.write(frame)

    cap.release()
    out.release()

import cv2
import time
import numpy as np
from numba import jit
from ultralytics import YOLO
from collections import defaultdict


class AccidentMath:

    @staticmethod
    @jit
    def get_distance(p1, p2):
        """
        :param p1: Coordinate of first point.
        :param p2: Coordinate of second point.
        :return: Distance between 2 points.
        """
        return np.linalg.norm(p2 - p1)
    

    @staticmethod 
    @jit
    def get_degree(px, py, fx, fy):
        """
        :param px: X coordinate of starting point.
        :param py: Y coordinate of starting point.
        :param fx: X coordinate of end point.
        :param fy: Y coordinate of end point.
        :return: Returns the angle of the line.
        """
        degree = np.degrees(np.arctan2(py - fy, px - fx))
        if degree < 0:
            return degree + 360
        else:
            return degree
        
    
    @staticmethod   
    @jit
    def get_direction(degree):
        """
        :param degree: Angle of line.
        :return: Returns the cos and sin value.
        """
        radi = np.radians(degree)
        c, s = np.cos(radi), np.sin(radi)
        return c, s
    

    @staticmethod 
    @jit
    def get_line(px, py, c, s, w, h, factor, back = False):
        """
        :param px: X coordinate of starting point.
        :param py: Y coordinate of starting point.
        :param c: Cos value.
        :param s: Sin value.
        :param w: Width of box.
        :param h: Height of box.
        :param factor: Specified length or width parameter.
        :param back: Direction control value.
        :return: Returns the start and end coordinates of the line.
        """
        uv = np.array([c, s]) * np.linalg.norm(np.array([w, h]) / 2 ) * factor
        if back:
            return (px, py, px + uv[0], py + uv[1])
        else:   
            return (px, py, px - uv[0], py - uv[1])
        

    @staticmethod      
    @jit
    def get_area(bx1, by1, bx2, by2, w, h, factor=1):
        """
        :param bx1: X coordinate of the starting point of the virtual area
        :param by1: Y coordinate of the starting point of the virtual area
        :param bx2: X coordinate of the end point of the virtual area
        :param by2: Y coordinate of the end point of the virtual area
        :param w: Width of box.
        :param h: Height of box. 
        :param factor: Specified length or width parameter.
        :return: Coordinates of the virtual area.
        """
        direction_vector = np.array([bx2 - bx1, by2 - by1]) 
        unit_vector = direction_vector / (np.linalg.norm(direction_vector) + 0.001)
        perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])

        unit_num = (np.sqrt(w**2 + h**2 + 0.001) / 4 * factor)

        a = (np.array([bx1, by1]) - unit_num * perpendicular_vector)
        b = (np.array([bx1, by1]) + unit_num * perpendicular_vector)
        c = (np.array([bx2, by2]) + unit_num * perpendicular_vector)
        d = (np.array([bx2, by2]) - unit_num * perpendicular_vector)
    
        return a, b, c, d
    

    @staticmethod 
    @jit
    def check_center(center, area):
        """
        :param center: Detector center point.
        :param area: Coordinates of the virtual area.
        :return: Virtual area control of the detector.
        """
        x, y = center
        n = len(area)
        check = False
        for i in np.arange(n):
            x1, y1 = area[i]
            x2, y2 = area[(i + 1) % n]
            if (y1 < y and y2 >= y) or (y2 < y and y1 >= y):
                if x1 + (y - y1) / (y2 - y1) * (x2 - x1) < x:
                    check = not check
        return check


    @staticmethod 
    @jit
    def get_border(x, y, w, degree):
        """
        :param x: X coordinate of border center.
        :param y: Y coordinate of border center.
        :param w: Width of border.
        :param degree: Angle of the border.
        :return: Start and end coordinates of the border line.
        """
        radi = np.radians(degree)

        sx = int(x - w * np.cos(radi) / 2)
        sy = int(y - w * np.sin(radi) / 2)
        fx = int(x + w * np.cos(radi) / 2)
        fy = int(y + w * np.sin(radi) / 2)

        return sx, sy, fx, fy
    

    @staticmethod 
    @jit
    def check_pos(sx, sy, fx, fy, spos, fpos):
        """
        :param sx: X coordinate of border starting point.
        :param sy: Y coordinate of border starting point
        :param fx: X coordinate of border end point.
        :param fy: Y coordinate of border end point.
        :param spos: Coordinate of the actual center of the vehicle.
        :param fpos: Coordinate of the future center of the vehicle
        :return: Border crossing control of the vehicle center.
        """
        m = (fy - sy) / (fx - sx)
        b = sy - m * sx
        x1, y1 = spos
        x2, y2 = fpos
        if ((y1 > m * x1 + b) and (y2 > m * x2 + b)) or ((y1 < m * x1 + b) and (y2 < m * x2 + b)):
            return "Not Accident"
        else:
            return "Accident"
        
    
    @staticmethod
    @jit
    def xywh2xyxy(x, y, w, h):
        """
        :param x: X coordinate of box center.
        :param y: Y coordinate of box center.
        :param w: Width of box.
        :param h: Height of box. 
        :return: Returns the top left and bottom right coordinates of the box.
        """
        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    

    @staticmethod
    @jit
    def boxes_concat(box1, box2):
        """
        :param box1: Coordinates of the box.
        :param box2: Coordinates of the box.
        :return: Returns the combined coordinates of two boxes.
        """
        x_min = int(min(box1[0], box2[0]))
        y_min = int(min(box1[1], box2[1]))
        x_max = int(max(box1[2], box2[2]))
        y_max = int(max(box1[3], box2[3]))
        return x_min, y_min, x_max, y_max
    

    @staticmethod
    @jit
    def get_factor(a_degree, c_degree, w, h):
        """
        :param a_degree: Direction angle of the virtual area.
        :param c_degree: Direction angle of the detector.
        :param w: Width of box.
        :param h: Height of box. 
        :return: Future vehicle center parameter.
        """
        if np.abs(a_degree - c_degree) < 60:
            return h / w
        else:
            return w / h
    

class AccidentDetection(AccidentMath):
    
    def __init__(self, source_path, model_path, save_path, video_save, past_num, future_num, threshold_distance, detector_h, area_h, area_w, verbose):
        super(AccidentMath, self).__init__()

        self.source_path = source_path
        self.model_path = model_path
        self.save_path = save_path
        self.video_save = video_save
        self.past_num = past_num
        self.future_num = future_num
        self.threshold_distance = threshold_distance
        self.detector_h = detector_h
        self.area_h = area_h
        self.area_w = area_w
        self.verbose = verbose

        self.img = None
        self.trackers = None
        self.degree = None
        self.state = "Not Accident"
        self.check_track = set()
        self.centers = defaultdict(list)
        self.areas = defaultdict(list)
        self.degrees = defaultdict(list)
        self.frame_ids = {}
        self.factor = {}
        self.times = []


    def draw(self):
        x1, y1, x2, y2 = np.int32([self.x1, self.y1, self.x2, self.y2])
        cv2.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(self.img, (x2, y2), 3, (0, 255,255), -1)
        cv2.polylines(self.img, np.int32([self.area]), isClosed=True, color=(255, 0, 0))
    

    def get_info(self):
        shapex = self.img.shape[1]
        shapey = self.img.shape[0]

        if self.state == "Accident":
            cv2.rectangle(self.img, (shapex-400, shapey-100), (shapex, shapey), (0, 0, 255), -1)
            cv2.putText(self.img, str(self.state), (shapex-300, shapey-25), 1, 3, (255, 255, 255))
        else:
            cv2.rectangle(self.img, (shapex-400, shapey-100), (shapex, shapey), (255, 0, 0), -1)
            cv2.putText(self.img, str(self.state), (shapex-350, shapey-25), 1, 3, (255, 255, 255))

    
    def time_write(self, process_time):
        self.times.append(process_time)
        if len(self.times) % 10 == 0:
            time_mean = sum(self.times) / 10
            self.times = []
            return time_mean
        

    def video_write(self):
        cap = cv2.VideoCapture(self.source_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(self.save_path, fourcc, fps, (width, height))
        return out
    
        
    @staticmethod
    def check_dist(c_car, a_car, c_center, a_center, c_degree, a_degree):
        """
        :param c_car: Tracker object of the detector.
        :param a_car: Tracker object of the virtual area.
        :param c_center: Center of detector.
        :param a_center: Center of virtual area.
        :param c_degree: Angle of the detector.
        :param a_degree: Angle of the virtual area.
        :return: Last position center and angle information is saved.
        """
        c_car.history["position"] = np.array(c_center)
        a_car.history["position"] = np.array(a_center)
        c_car.history["last_degree"] = c_degree
        a_car.history["last_degree"] = a_degree
        return c_car, a_car
    
    
    def calculate_process(self):

        for i in range(len(self.result.boxes.xywh)):

            x, y, w, h = self.result.boxes.xywh[i].numpy()

            track= self.trackers.tracked_stracks[i]
            track_id = self.trackers.tracked_stracks[i].track_id
                        
            self.trackers.tracked_stracks[i].features.append(np.array((x, y)))

            if len(self.trackers.tracked_stracks[i].features) > self.past_num:

                ox, oy = self.trackers.tracked_stracks[i].features[-self.past_num]

                dist = self.get_distance(np.array((ox, oy)), np.array((x, y)))

                if dist > self.threshold_distance:
                    degree = self.get_degree(ox, oy, x, y)
                    self.trackers.tracked_stracks[i].history["degree"] = degree

                elif self.trackers.tracked_stracks[i].history:
                    degree = self.trackers.tracked_stracks[i].history["degree"]
                
                else:
                    continue

                c, s = self.get_direction(degree)
                
                x1, y1, x2, y2 = self.get_line(x, y, c, s, w, h, factor=self.detector_h, back=False)

                _, _, bx1, by1 = self.get_line(x, y, c, s, w, h, factor=self.area_h, back=True)
                _, _, bx2, by2 = self.get_line(x, y, c, s, w, h, factor=self.area_h, back=False)

                area = self.get_area(bx1, by1, bx2, by2, w, h, factor=self.area_w)

                self.areas[track_id].append((track, area, x, y, w, h, self.result.boxes.xyxy[i].numpy(), degree))
                self.centers[track_id].append((track, x1, y1, (x2, y2), c, s, self.result.boxes.xyxy[i].numpy(), degree))
                self.degrees[track_id].append(degree)

                self.x1, self.y1, self.x2, self.y2, self.area, self.degree = x1, y1, x2, y2, area, degree

                self.draw()


            self.degree = None


    def match_process(self):
        
        for area_id, area_info in self.areas.items():
            for center_id, center_info in self.centers.items():  

                if area_id == center_id:
                    continue
            
                (c_track, x1, y1, center, c, s, c_xyxy, c_degree) = center_info[-1]
                (a_track, area, x, y, w, h, a_xyxy, a_degree) = area_info[-1]

                check = self.check_center(center, area)

                if check:
                    
                    # img = draw_car_area(img, area, (-50, -50, 255))

                    degree = c_track.history["degree"] - 90

                    sx, sy, fx, fy = self.get_border(x, y, w, degree)

                    cv2.line(self.img, (sx, sy), (fx, fy), (0, 255, 0), 3, 2)
                    
                    # trackers.multi_predict(trackers.tracked_stracks)

                    mean = c_track.mean[4:6]

                    spos = int(x1), int(y1)

                    if not(area_id in self.factor.keys()):

                        self.factor[area_id] = self.get_factor(a_degree, c_degree, w, h)

                    fpos = int(x1 + (self.future_num * mean[0] * self.factor[area_id])), int(y1 + (self.future_num * mean[1] * self.factor[area_id]))

                    cv2.circle(self.img, (fpos[0], fpos[1]), 10, (0, 0,255), -1)

                    self.state = self.check_pos(sx, sy, fx, fy, spos, fpos)

                    if self.state == "Accident":

                        bx1, by1, bx2, by2 = self.boxes_concat(a_xyxy, c_xyxy)
                        cv2.putText(self.img, "Accident", (bx1, by1- 20), 3, 1, (0, 0, 255))
                        cv2.rectangle(self.img, (bx1, by1), (bx2, by2), (0, 0, 255), 5)

                        if "position" in c_track.history.keys():
                            continue

                        self.frame_ids[c_track.track_id] = c_track.frame_id
                        self.check_track.add(self.check_dist(c_track, a_track, (x1, y1), (x, y), c_degree, a_degree))     
                    
                else:
                    self.state = "Not Accident"
                    # img = draw_car_area(img, area, (255, -50, -50))


    def check_process(self):

        for tracked in self.check_track:
        
            cx, cy = tracked[0].history["position"]
            ax, ay = tracked[1].history["position"]

            if (not(tracked[0] in self.trackers.removed_stracks) and self.get_distance(np.array((cx, cy)), tracked[0].features[-1]) < 1000):
                self.state = "Accident"

            if (not(tracked[0] in self.trackers.removed_stracks) and self.get_distance(np.array((ax, ay)), tracked[1].features[-1]) < 1000):
                self.state = "Accident"

            if (self.trackers.tracked_stracks[0].frame_id - self.frame_ids[tracked[0].track_id] > 20):
                continue

            c_state = (np.abs(tracked[0].history["last_degree"] - self.degrees[tracked[0].track_id][-1]) > 30)
            a_state = (np.abs(tracked[1].history["last_degree"] - self.degrees[tracked[1].track_id][-1]) > 30)

            if c_state:
                self.state = "Accident"
                x, y, w, h = tracked[0].mean[:4]
                x1, y1, x2, y2 = self.xywh2xyxy(x, y, w, h)
                cv2.putText(self.img, "Direction change detected.", (int(x1), int(y1 - 20 )), 3, 1, (1))
                cv2.circle(self.img, (int(x), int(y)), 20, (0, 255, 255), 3)

            if a_state:
                self.state = "Accident"
                x, y, w, h = tracked[1].mean[:4]
                x1, y1, x2, y2 = self.xywh2xyxy(x, y, w, h)
                cv2.putText(self.img, "Direction change detected.", (int(x1), int(y1 - 20 )), 3, 1, (1))
                cv2.circle(self.img, (int(x), int(y)), 20, (0, 255, 255), 3)
                

    def detect(self):
        
        out = self.video_write()

        model = YOLO(self.model_path)

        for i, result in enumerate(model.track(source=self.source_path, stream=True, imgsz=(736, 608), vid_stride=1, classes=[2], conf=0.7, persist=True, verbose=False)):

        
            self.img = result.plot(conf=False, labels=True, line_width=1)
            
            t = time.time()

            self.trackers = model.predictor.trackers[0]
            self.result = result

            # PROCESS 1
            self.calculate_process()
            
            # PROCESS 2
            self.match_process()
            
            # PROCESS 3
            self.check_process()
                

            if self.verbose:
                time_mean = self.time_write(process_time=time.time() - t)
                if time_mean:
                    print(f"{i - 9}-{i + 1}.Frame Average Processing Time: {time_mean * 1000 :.2f} ms") 

            self.get_info()
            
            cv2.imshow("", self.img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                    break  
            
            if self.video_save:
                out.write(self.img)

        cv2.destroyAllWindows()
        out.release()
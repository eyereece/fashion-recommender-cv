import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class ObjDetection():
    def __init__(self, onnx_model, data_yaml):
        # load data yaml
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # load object detection model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def crop_objects(self, image):
        row, col, d = image.shape

        # convert img into square array
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward() # prediction from model

        # NMS
        # filter detection based on conf (0.1) and prob (0.1) score
        # initialize
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # width and height of the image (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4] # confidence of obj detection
            if confidence > 0.10:
                class_score = row[5:].max() # max probability from all objects
                class_id = row[5:].argmax() # get that index position

                if class_score > 0.10:
                    cx, cy, w, h = row[0:4]
                    
                    # construct bbox from 4 values
                    # left, top, width and height
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])

                    # append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # clean up
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # obtain NMS
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.10, 0.10).flatten()

        # obtain bbox and crop object images
        cropped_objects = []
        for ind in index:
            x, y, w, h = boxes_np[ind]
            x1 = int(x)
            y1 = int(y)
            x2 = int((x + w))
            y2 = int((y + h))

            # crop item from image
            cropped_obj = image[y1:y2, x1:x2].copy()
            cropped_objects.append(cropped_obj)

        return cropped_objects

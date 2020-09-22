import cv2,os,urllib.request
import numpy as np
from django.conf import settings

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils

# load our serialized face detector model from disk
# prototxtPath = os.path.join("face_detector/deploy.prototxt")
# weightsPath = os.path.join("face_detector/res10_300x300_ssd_iter_140000.caffemodel")
# print(prototxtPath)

# faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
# # maskNet = load_model(os.path.join('face_detector/mask_detector.model'))
# print(faceNet)
os.chdir(r'C:/Users/rjsgh/OneDrive/바탕 화면/Mask_Detection/warming_up_project/streamapp')

weightsPath = "./face_detector/yolov4-tiny-obj_best.weights"
cfgPath = "./face_detector/yolov4-tiny-obj.cfg"
yoloNet = cv2.dnn.readNet(weightsPath, cfgPath)

layer_names = yoloNet.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yoloNet.getUnconnectedOutLayers()]

LABELS = ["Mask", "No Mask"]
COLORS = [(0,255,0), (0,0,255)]

class yoloDetect:
    def __init__(self):
        self.vs = VideoStream(src=0).start()
        # self.vs = cv2.VideoCapture(0)
    
    def __del__(self):
        # self.vs.release()
        cv2.destroyAllWindows()
        self.vs.stop()


    def detect_and_predict_mask(self, frame):
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        yoloNet.setInput(blob)
        layerOutputs = yoloNet.forward(output_layers)
        
        class_ids = []
        cofidences = []
        boxes = []

        for output in layerOutputs :
            for detection in output :
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 :
                    box = detection[:4] * np.array([w,h,w,h])
                    (center_x, center_y, width, height) = box.astype("int")

                    x = int(center_x - width/2)
                    y = int(center_y - height/2)

                    boxes.append([x,y,int(width),int(height)])
                    cofidences.append(float(confidence))
                    class_ids.append(class_id)

        return class_ids, cofidences, boxes

    def get_frame(self):

        frame = self.vs.read()
        # _, frame = self.vs.read()
        frame = imutils.resize(frame, width=650)
        frame = cv2.flip(frame, 1)
        class_ids, cofidences, boxes = self.detect_and_predict_mask(frame)

        # 같은 물체에 대한 박스가 많은 것을 제거
        # YOLO does not apply non-maxima suppression for us, so we need to explicitly apply it.
        idxs = cv2.dnn.NMSBoxes(boxes, cofidences, 0.5, 0.3)

        if len(idxs) > 0 :
            for i in idxs.flatten() :
                x, y, w, h = boxes[i]
                label = str(LABELS[class_ids[i]])
                color = COLORS[class_ids[i]]
                label = "{}: {:.2f}%".format(label, cofidences[i] * 100)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # print(color[1], color[2])
                # No Mask: 93.99%
                # (0, 0, 255) or (0, 255, 0)

                # if color[1] == 255:
                #     cnt += 1
                # if cnt > 20:
                #     cnt = 0
                #     # arduino.write(b'y')
                #     time.sleep(5.5)
                    
                # print(cnt)

        ret, jpeg = cv2.imencode('.jpg', (frame))
        return jpeg.tobytes()

# class MaskDetect(object):
#     # threaded video stream을 시작한다.
# 	def __init__(self):
# 		self.vs = VideoStream(src=0).start()
    
#     # __del__ -> finalizer
# 	def __del__(self):
#         # 우리가 만든 창 다 닫는 것
# 		cv2.destroyAllWindows()

# 	def detect_and_predict_mask(self, frame, faceNet, maskNet):
# 		# grab the dimensions of the frame and then construct a blob
# 		# from it
# 		(h, w) = frame.shape[:2]
# 		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
# 									 (104.0, 177.0, 123.0))

# 		# pass the blob through the network and obtain the face detections
# 		faceNet.setInput(blob)
# 		detections = faceNet.forward()

# 		# initialize our list of faces, their corresponding locations,
# 		# and the list of predictions from our face mask network
# 		faces = []
# 		locs = []
# 		preds = []

# 		# loop over the detections
# 		for i in range(0, detections.shape[2]):
# 			# extract the confidence (i.e., probability) associated with
# 			# the detection
# 			confidence = detections[0, 0, i, 2]

# 			# filter out weak detections by ensuring the confidence is
# 			# greater than the minimum confidence
# 			if confidence > 0.5:
# 				# compute the (x, y)-coordinates of the bounding box for
# 				# the object
# 				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 				(startX, startY, endX, endY) = box.astype("int")

# 				# ensure the bounding boxes fall within the dimensions of
# 				# the frame
# 				(startX, startY) = (max(0, startX), max(0, startY))
# 				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

# 				# extract the face ROI, convert it from BGR to RGB channel
# 				# ordering, resize it to 224x224, and preprocess it
# 				face = frame[startY:endY, startX:endX]
# 				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
# 				face = cv2.resize(face, (224, 224))
# 				face = img_to_array(face)
# 				face = preprocess_input(face)

# 				# add the face and bounding boxes to their respective
# 				# lists
# 				faces.append(face)
# 				locs.append((startX, startY, endX, endY))

# 		# only make a predictions if at least one face was detected
# 		if len(faces) > 0:
# 			# for faster inference we'll make batch predictions on *all*
# 			# faces at the same time rather than one-by-one predictions
# 			# in the above `for` loop
# 			faces = np.array(faces, dtype="float32")
# 			preds = maskNet.predict(faces, batch_size=32)

# 		# return a 2-tuple of the face locations and their corresponding
# 		# locations
# 		return (locs, preds)

# 	def get_frame(self):
#         # 현재 프레임을 가져온다
# 		frame = self.vs.read()
# 		frame = imutils.resize(frame, width=650)
# 		frame = cv2.flip(frame, 1)
# 		# detect faces in the frame and determine if they are wearing a
# 		# face mask or not
# 		(locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

# 		# loop over the detected face locations and their corresponding
# 		# locations
# 		for (box, pred) in zip(locs, preds):
# 			# unpack the bounding box and predictions
# 			(startX, startY, endX, endY) = box
# 			(mask, withoutMask) = pred

# 			# determine the class label and color we'll use to draw
# 			# the bounding box and text
# 			label = "Mask" if mask > withoutMask else "No Mask"
# 			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

# 			# include the probability in the label
# 			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

# 			# display the label and bounding box rectangle on the output
# 			# frame
# 			cv2.putText(frame, label, (startX, startY - 10),
# 						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
# 			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

#         ret, jpeg = cv2.imencode('.jpg', frame)
#         return jpeg.tobytes()
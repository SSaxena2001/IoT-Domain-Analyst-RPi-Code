import cv2
import mediapipe as mp
import time
import os

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.5)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.pTime = 0
        self.cTime = 0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, img = self.video.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)

        if results.detections:
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(img, detection)
                # print(id, detection)
                # print(detection.location_data.relative_bounding_box)
                bounding_box_c = detection.location_data.relative_bounding_box
                ih, iw, c = img.shape
                bounding_box = int(bounding_box_c.xmin * iw), int(bounding_box_c.ymin * ih), \
                               int(bounding_box_c.width * iw), int(bounding_box_c.height * ih)
                cv2.rectangle(img, (bounding_box[0], bounding_box[1]),(bounding_box[2], bounding_box[3]), (255, 0, 255), 2)
                cv2.putText(img, f"{int(detection.score[0] * 100)}%",
                            (bounding_box[0], bounding_box[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0), 1
                            )

        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime

        cv2.putText(img, f"FPS: {int(fps)}",
                    (20, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 0, 0), 1
                    )
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

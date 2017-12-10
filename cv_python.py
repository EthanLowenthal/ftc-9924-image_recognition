import numpy as np
import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils

center = cv2.CascadeClassifier('opencv-center-classifier/classifier/cascade.xml')
right = cv2.CascadeClassifier('opencv-right-classifier/classifier/cascade.xml')
left = cv2.CascadeClassifier('opencv-left-classifier/classifier/cascade.xml')

#img = cv2.imread('sachin.jpg')
video_capture = WebcamVideoStream(src=0).start()
fps = FPS().start()
frame = video_capture.read()
#ret, frame = video_capture.read()
width = np.size(frame, 1)

classifiers = [(center,"Center",(255,0,0)),(right,"Right",(0,255,0)),(left,"left",(0,0,255))]

while True:
	detections = []
	#time.sleep(0.05)
	#ret, frame = video_capture.read()
	frame = video_capture.read()
	fframe = cv2.flip( frame, 1 )
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	for classifier in classifiers:
		detection = classifier[0].detectMultiScale(gray,scaleFactor=1.1,
        	minNeighbors=5,minSize=(30, 30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
		
		if type(detection) is not tuple:
			detections.append([detection, classifier[1], classifier[2]])

	
	for d in detections:
		for (x,y,w,h) in d[0]:

			cv2.rectangle(fframe,(width-x,y),(width-x+(w/2),y+h),d[2],2)
			ty = y - 15

			cv2.putText(fframe, d[1], (width-x, ty),cv2.FONT_HERSHEY_SIMPLEX, 1, d[2], 				thickness=3)
			

	cv2.imshow('FTC image recognition',fframe)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break

	fps.update()


#cv2.waitKey(0)
fps.stop()
video_capture.release()
cv2.destroyAllWindows()


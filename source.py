#import libraries
from imutils.video import VideoStream
import datetime 
import imutils 
import time 
import cv2

#initialize values
T = 50 #image binarization
min_area = 1000 #minimum contour area to insignificant contour suppression
background = None #store the background

#if you would like to use pre-recorded video, you should provide the path to the video location. if you would like to use the camera, video_path variable should remain None
video_path = None
if video_path is None:  
	vs = VideoStream().start()  
	#warm up the camera 
	time.sleep(2)  
else:   
	vs = cv2.VideoCapture(video_path)

#read images
while True:  
	frame = vs.read() 
	frame = frame if video_path is None else frame[1]

	#keeps track of the state of motion detection 
	state = “No change”

	#if there is no more frame in the video leave the loop 
	if frame is None:  
		break
	
	#resize frame, convert it to grey scale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#smooting image in 21x21 pixels regions 
	gray = cv2.GaussianBlur(gray, (21,21), 0)

	#if the background is None, initialize it
	if background is None:           
		background = gray.copy().astype("float")           
		continue

	#compute the absolute difference between the current frame and the background
	delta_frame = cv2.absdiff(gray, cv2.convertScaleAbs(background))

	#delta frame goes through a threshold function where less significant changes on the image will be suppressed
	threshold = cv2.threshold(delta_frame, T, 255, cv2.THRESH_BINARY)[1]
	threshold = cv2.dilate(threshold, None, iterations=2)

	#apply contour detection to find outlines of white region (moving object)
	cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
	cnts = imutils.grab_contours(cnts)


	#contours will be taken consider if they have greater area than the minimum value. if true, the program draws a bounding box around the contour and the motion state will be changed
	for c in cnts:    
		if cv2.contourArea(c) < min_area:     
			continue        
		(x,y,w,h) = cv2.boundingRect(c)    
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		state = "New object"
	
	#accumulate the weighted average between the current frame and earlier frames
	cv2.accumulateWeighted(gray, background, 0.5)

	#visualize result to user
	cv2.putText(frame, "Room Status: {}".format(state), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	cv2.imshow("Camera image", frame)
	cv2.imshow("Threshold", threshold)
	cv2.imshow("Delta frame", delta_frame)
	key = cv2.waitKey(1) & 0xFF

	#q key to stop the algorithm
	if key == ord("q"): 
		break  

#shut down the camera and close all open windows
vs.stop() if video_path is None else vs.release()
cv2.destroyAllWindows()
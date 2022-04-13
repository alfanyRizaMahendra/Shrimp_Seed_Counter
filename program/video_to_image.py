import cv2
import time

cap = cv2.VideoCapture('video2.mp4')
    
# FRAMES PER SECOND FOR VIDEO
# fps = 33

index = 49

if cap.isOpened()== False: 
	print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")

# While the video is opened
while cap.isOpened():	
	# Read the video file.
	ret, frame = cap.read()
	# If we got frames show them.
	if ret == True:
		# time.sleep(1/fps)
		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('s'):
			filename = 'PL3_'+ str(index) + '.jpg'
			frame = frame[85:985, 490:1390,:]
			print(frame.shape)
			cv2.imwrite(filename, frame)
			index = index + 1

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# Or automatically break this whole loop if the video is over.
	else:
		break
				
cap.release()
cv2.destroyAllWindows()
import tensorflow as tf
from collections import deque
import numpy as np
import argparse
import cv2


input_video='flood.mp4'
output_video='output.avi'
display=1
size=128
CLASSES = ["Cyclone", "Earthquake", "Flood", "Wildfire"]


print("[INFO] loading model and label binarizer...")
model= tf.keras.models.load_model('model.hdf5')

Q = deque(maxlen=size)

print("[INFO] processing video...")
vs = cv2.VideoCapture(input_video)
writer = None
(W, H) = (None, None)


# loop over the video
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	# frame dimensions are empty, grab them
	if W is None or H is None:
		    (H, W) = frame.shape[:2]


	# clone the output frame, then convert it from BGR to RGB
	# ordering and resize the frame to a fixed 224x224
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224))
	frame = frame.astype("float32")

	#prediction
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = CLASSES[i]

	# write on the output frame
	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output_video, fourcc, 30,(frame.shape[1], frame.shape[0]),True)

	# write the output
	writer.write(output)


	if display > 0:
		# show the output image
		cv2.imshow("Output", output)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

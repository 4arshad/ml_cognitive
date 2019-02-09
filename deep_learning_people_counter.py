# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=False, default=0,
    help="path to input image")
ap.add_argument("-p", "--prototxt", required=False, default="deps\MobileNetSSD_deploy.prototxt.txt",
    help="path to Caffe 'deploy' prototxt file",)
ap.add_argument("-m", "--model", required=False, default="deps\MobileNetSSD_deploy.caffemodel",
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

cap = cv2.VideoCapture(1)

cap.set(3,1280)
cap.set(4,720)

process_one_every_n_frame = 5
curr_frame = 1

while True:
    try:
        cap.grab()
        if (curr_frame % process_one_every_n_frame == 0):
            curr_frame = 1
        else:
            curr_frame = curr_frame + 1
            continue
        ret, image = cap.retrieve()
    except:
        print("Error: Restarting capture")
        cap = cv2.VideoCapture(args["video"])
        continue
    
    if image is None:
        print("Error: Restarting capture")
        cap = cv2.VideoCapture(args["video"])
        continue

    # image = cv2.resize(image, (1280, 720))
    (h, w) = image.shape[:2]
    # blob = cv2.dnn.blobFromImage(cv2.resize(image, (400, 400)), 0.007843, (400, 400), 127.5)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    people_count = 0
    
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            if ( idx == 15 ):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))

                # if ( idx == 15 ):
                people_count = people_count + 1

                cv2.rectangle(image, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 2)

    # show the output image
    #image = cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
    cv2.putText(image, 'People count: {}'.format(people_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 128), 2)
    
    # cv2.imshow("Output", cv2.resize(image, (3640, 2060)))
    cv2.imshow("Output", image)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        
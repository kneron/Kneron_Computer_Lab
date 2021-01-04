# Name      : yolo3.py
# Author    : Oscar Law
#
# Decription:
#
#   Implement yolo3 algorithm for image/video detection
#
# History:
# Jul 04, 2019. O. Law
# - created

import numpy as np
import argparse
import time
import cv2
import os
import sys

def getModel():

    yoloLabels  = 'common\yolo-coco\coco.names'
    yoloConfig  = 'common\yolo-coco\yolov3.cfg'
    yoloWeights = 'common\yolo-coco\yolov3.weights'

    labelsPath  = os.path.join(os.getcwd(), yoloLabels)
    labels      = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    colors      = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    configPath  = os.path.join(os.getcwd(), yoloConfig)
    weightsPath = os.path.join(os.getcwd(), yoloWeights)

    net         = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln          = net.getLayerNames()
    ln          = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return(net, ln, labels, colors)

def detectImage():

    yoloImages       = 'images\\'
    imagePath       = os.path.join(os.getcwd(), yoloImages)

    imageFlag       = True
    yoloConfidence  = 0.5
    yoloThreshold   = 0.3

    net, ln, labels, colors = getModel()

    imageFile   = input('Input image file: ')
    imageFile   = os.path.join(imagePath, imageFile)
    imageFlag   = os.path.isfile(imageFile)

    while imageFlag:
        font_size = 0.5
        boldface = 2
        image = cv2.imread(imageFile)
        (H, W) = image.shape[:2]
        font_size *= W/600
        boldface = int(boldface * W/600)

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if (confidence > yoloConfidence):
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, yoloConfidence, yoloThreshold)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, boldface)
                print ("{}: {:.4f} {:d},{:d} {:d},{:d}".format(labels[classIDs[i]], confidences[i], x, y, x + w, y + h))
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, boldface)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        cv2.putText(image, "press key 'n' to enter next picture's name ", (0,int(H*0.9)), cv2.FONT_HERSHEY_PLAIN, 3*font_size, (0, 0, 50), 1, 20*boldface)
        cv2.putText(image, "press 'q' to leave", (0,H), cv2.FONT_HERSHEY_PLAIN, 3*font_size, (0, 0, 50), 1, 20*boldface)
        cv2.resizeWindow("Image", (600, 450))
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            del image
            break
        elif key == ord('n'):
            del image

        imageFile   = input('Input image file: ')
        imageFile   = os.path.join(imagePath, imageFile)
        imageFlag   = os.path.isfile(imageFile)

    return()

def detectVideo():

    yoloConfidence  = 0.5
    yoloThreshold   = 0.3

    net, ln, labels, colors = getModel()

    cam     = cv2.VideoCapture(0)
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", (600, 450))
    (W, H) = (None, None)

    while True:

        success, frame = cam.read()
#       cv2.imshow("Video", frame)

        if not success:
            break
        key = cv2.waitKey(5)

        if (key % 256 == 27):
            print ("End program ...")
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        frame = cv2.flip(frame,1)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > yoloConfidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, yoloConfidence, yoloThreshold)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, "press key 'q' to leave", (410,470), cv2.FONT_HERSHEY_PLAIN, 1.2, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow("Video", frame)
        #cv2.waitKey(5)
        if key == ord('q'):
            sys.exit()

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ### input parameters ###
    argparser = argparse.ArgumentParser(
        description="Run Yolov3 examples by calling GPU calculation",
        formatter_class=argparse.RawTextHelpFormatter)

    argparser.add_argument(
        '-t',
        '--task_name',
        help=("image\ncamera"), default="camera")

    args = argparser.parse_args()


    print("Task: ", args.task_name)
    ### parse parameters and run different example ###
    {
        "image": detectImage,
        "camera": detectVideo,
    }.get(args.task_name, lambda: 'Invalid test')()





#detectImage()
#detectVideo()

import cv2
import numpy as np
import os, sys, inspect #For dynamic filepaths
import random


# Pretrained classes in the model - Dictionary
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

#Function to return name from the dictionary
def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

#Find the execution path and join it with the direct reference
def execution_path(filename):
  return os.path.join(os.path.dirname(inspect.getfile(sys._getframe(1))), filename)			

#Loading model
model = cv2.dnn.readNetFromTensorflow(execution_path('models/frozen_inference_graph.pb'),
                                      execution_path('models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'))
cam = cv2.VideoCapture(0)


while True:

    check, frame = cam.read()
    image = cv2.resize(frame, (320,280))
    
    #Greyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Threshold 120 is threshold, 255 is what we assign if it is below this
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    image = cv2.blur(image, (5,5))

    #Canny
    image = cv2.Canny(image, 350,400)

    #Countours (needs canny)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print("Number of Contours Found = " + str(len(contours)))
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, contours, -1, (255,255,255),2)

    # image1 = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
    image1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lines = cv2.HoughLinesP(image1, 1, np.pi/180.0, 100, minLineLength=100, maxLineGap=10)
    image2 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    parallel_lines = {}
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            slope = (y2-y1)/(x2-x1) if (x2-x1) != 0 else float('inf')
            key = round(slope, 2)
            if key not in parallel_lines:
                parallel_lines[key] = []
            parallel_lines[key].append((x1, y1, x2, y2))
        for slope, line_group in parallel_lines.items():
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            for x1, y1, x2, y2 in line_group:
                image3 = cv2.line(image2, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.putText(image3, f'Slope: {slope}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imshow('image', image3)
    else:
        cv2.imshow('image', image2)

    # lines = cv2.HoughLines(image1, 1, math.pi/180.0,100, np.array([]), 0, 0)
    # image2 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    # if lines is not None:
    #     a,b,c = lines.shape
    #     for i in range(a):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         image3 = cv2.line(image2, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)

    # lines = cv2.HoughLines(image2, 1, np.pi/180, 200)
    # print (lines)
    # if lines is not None:
    #     for r_theta in lines:
    #         arr = np.array(r_theta[0], dtype=np.float64)
    #         r,theta = arr
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*r
    #         y0 = b*r
    #         x1 = int(x0 + 1000*(-b))
    #         y1 = int(y0 + 1000*(a))
    #         x2 = int(x0 - 1000*(-b))
    #         y2 = int(y0 - 1000*(a))
    #         #image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    #         image3 = cv2.line(image2, (x1,y1), (x2,y2), (0,0,255), 2, cv2.LINE_AA)
    # for i in contours:
    #     M = cv2.moments(i)
    #     if M['m00'] != 0:
    #         cx = int(M['m10']/M['m00'])
    #         cy = int(M['m01']/M['m00'])
    #         #cv2.drawContours(image, [i], -1, (0,255,0), 2)
    #         cv2.circle(image, (cx,cy), 7, (0,255,0), -1)
    #         cv2.putText(image, 'center', (cx-20,cy-20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)

    image_height, image_width = image.shape

    #Sets our input as the image, turns it into a blob
    #Resizes and sets the colour mode to BGR
   # model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))

    # #Returns a blob array
    # output = model.forward()



    # for detection in output[0, 0, :, :]:
    #     confidence = detection[2]
    #     if confidence > .5: #This is our confidence threshold
    #         class_id = detection[1] #This is the ID of what it thinks it is
    #         class_name=id_class_name(class_id,classNames) #Returning the name from Dictionary
    #         print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
            
    #         #Draw the bounding box, scaled to size of the image
    #         box_x = detection[3] * image_width
    #         box_y = detection[4] * image_height
    #         box_width = detection[5] * image_width
    #         box_height = detection[6] * image_height
    #         cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=2)
            
    #         #Put some text on the bounding box
    #         cv2.putText(image,class_name ,(int(box_x), int(box_y+.0001*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.004*image_width),(0, 0, 255), thickness=2)


#cv2.imwrite("image_box_text.jpg",image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()

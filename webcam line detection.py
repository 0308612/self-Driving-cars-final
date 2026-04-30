import cv2
import numpy as np
import os, sys, inspect #For dynamic filepaths
import random
import itertools
import warnings


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
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)


while True:

    check, frame = cam.read()
    image = cv2.resize(frame, (640,480))

    #Greyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Threshold 120 is threshold, 255 is what we assign if it is below this
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    image = cv2.blur(image, (5,5))

    #Canny
    image = cv2.Canny(image, 350,400,3)

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
        image4 = None
        for line in lines:
            x1,y1,x2,y2 = line[0]
        #    slope = (y2-y1)/(x2-x1) if (x2-x1) != 0 else float('inf')
        #     key = round(slope, 1) # Group lines by slope rounded to 1 decimal place
        #     if key not in parallel_lines:
        #         parallel_lines[key] = []
        #     parallel_lines[key].append((x1, y1, x2, y2))
        # for slope, line_group in parallel_lines.items():
        #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        #     for x1, y1, x2, y2 in line_group:
        #         image3 = cv2.line(image2, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        #         cv2.putText(image3, f'Slope: {slope}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            midlines = []
            midlines.append(((x1+x2)/2, (y1+y2)/2))
        pts = np.array(midlines, dtype=np.float32)
        if 'pts' in locals() and pts.size >= 2:
            pts = pts[pts[:, 0].argsort()]  # Sort points by x-coordinate
            x_data = pts[:, 0]
            y_data = pts[:, 1]
            
            deg = 2 if pts.size > 5 else 1  # Use quadratic fit for more points, linear fit for fewer
            z = np.polyfit(y_data, x_data, deg)  # Fit a quadratic polynomial
            p = np.poly1d(z)

            y_range = np.linspace(y_data.min(), y_data.max(), 50)
            x_range = p(y_range)

            curve_pts = np.stack([x_range, y_range], axis=-1).astype(np.int32)
            curve_pts = curve_pts.reshape((-1, 1, 2))
            cv2.polylines(image2, [curve_pts], isClosed=False, color=(0, 255, 0), thickness=2)
        else:
            print("Not enough points to fit a curve.")

            # vx,vy,x,y = [v.item() for v in cv2.fitLine(pts, cv2.DIST_L2,0,0.01,0.01)]
            # print(f"Line direction vector: ({vx}, {vy}), Point on line: ({x}, {y})")

            # x_coords = pts[:, 0]
            # y_coords = pts[:, 1]

            # z = np.polyfit(x_coords, y_coords, 2)
            # p = np.poly1d(z)

            # plot_x = np.linspace(min(x_coords), max(x_coords), 100).astype(int)
            # plot_y = p(plot_x).astype(int)

            # curve_pts = np.column_stack((plot_x, plot_y))
            # cv2.polylines(image2, [curve_pts], isClosed=False, color=(0, 255, 0), thickness=2)


        
        # most_parallel_pair = None
        # exact_parallel_cutoff = 1e-1000 
        # min_diff = float('inf')
        # diff = abs(np.arctan(slope) - np.arctan(slope))  # This will be zero for lines with the same slope
        # for l1, l2 in itertools.combinations(lines, 2):
        #     diff = np.minimum(diff, np.pi - diff)
        #     if exact_parallel_cutoff < diff < min_diff:
        #         min_diff = diff
        #         most_parallel_pair = (l1, l2)
        #         image4 = cv2.line(image2, (l1[0][0], l1[0][1]), (l1[0][2], l1[0][3]), (0, 255, 0), 3, cv2.LINE_AA)
        #         image4 = cv2.line(image2, (l2[0][0], l2[0][1]), (l2[0][2], l2[0][3]), (0, 255, 0), 3, cv2.LINE_AA)

        
        if image4 is not None:
            cv2.imshow('image', image4)
        else:
            cv2.imshow('image', image2)
    else:
        cv2.imshow('image', image2)


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

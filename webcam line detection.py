import cv2
import numpy as np
import warnings
import serial
import time

cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()
    x_range = []
else:
    ser = None

while True:
    if ser is not None:
        for i in range(len(x_range)):
            if x_range[i] != 0:
                print(f"Sending x_range: {x_range[i]}")
                ser.write(str.encode(str(x_range[i])))
                i += 1


    check, frame = cam.read()
    image = cv2.resize(frame, (640,480))

    #Greyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Threshold 120 is threshold, 255 is what we assign if it is below this
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_OTSU)

    #image = cv2.blur(image, (5,5))

    image = cv2.GaussianBlur(image, (5,5), 0)

    image = cv2.bilateralFilter(image, 9, 75, 75)

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

    warnings.filterwarnings("ignore", category=np.exceptions.RankWarning)  # Suppress RankWarning from polyfit
    if lines is not None:
        midlines = []
        image4 = None
        for line in lines:
            x1,y1,x2,y2 = line[0]

            midlines.append(((x1+x2)/2, (y1+y2)/2))
            #print(str(midlines) + " midlines")
        pts = np.array(midlines, dtype=np.float32)
        #print(pts.shape[0])
        if 'pts' in locals() and pts.size > 2:
            pts = pts[pts[:, 0].argsort()]  # Sort points by x-coordinate
            x_data = pts[:, 0]
            y_data = pts[:, 1]
            #print(str(x_data) + " x data, " + str(y_data) + " y data")
        
            deg = 2 if pts.size > 5 else 1  # Use quadratic fit for more points, linear fit for fewer
            z = np.polyfit(x_data, y_data, deg)  # Fit a quadratic polynomial
            p = np.poly1d(z)

            #print (str(np.min(x_data)) + " min x data, " + str(np.max(x_data)) + " max x data")
            x_range = np.linspace(x_data.min(), x_data.max())
            #print(str(x_range) + " x range")
            y_range = p(x_range).astype(np.int32)

            curve_pts = np.stack([y_range, x_range], axis=-1).astype(np.int32)
            curve_pts = curve_pts.reshape((-1, 1, 2))
            #print(str(curve_pts) + " curve pts")
            cv2.polylines(image2, [curve_pts], isClosed=False, color=(0, 255, 0), thickness=2)
        else:
            print("Not enough points to fit a curve.")

    
        if image4 is not None:
            cv2.imshow('image', image4)
        else:
            cv2.imshow('image', image2)
    else:
        cv2.imshow('image', image2)

    image_height, image_width = image.shape


    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()

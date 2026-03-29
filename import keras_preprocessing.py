import keras_preprocessing 
from keras_preprocessing import image
import cv2
import matplotlib.pyplot as plt

img = image.load_img("city.jpeg", grayscale=True, target_size=(480, 640))
img = image.img_to_array(img, dtype= 'uint8')

print(img.shape)

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
plt.figure(figsize=(20,10))
plt.imshow(th3, cmap='gray')
plt.show()
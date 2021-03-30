
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
import math


model = keras.models.load_model("lane_navigation.h5")

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
    image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
    return image

  
def compute_steering_angle(frame):
    preprocessed = img_preprocess(frame)
    X = np.asarray([preprocessed])
    steering_angle = model.predict(X)[0]
    return steering_angle

# to make coordinates from the slope and the intercept (biase)
def make_coordinates( image, line_parameters ):
    slope = line_parameters
    intercept = -10
    # to get image height
    y1 = image.shape[0]
    # we want the line to start from the buttom to 3/5 from the image height
    y2 = int( y1 * ( 3/5 ))
    # simple equation to get x1 and x2 dependent on the y and slope and intercept (bias)
    x1 = int(( y1 - intercept ) / slope )
    x2 = int(( y2 - intercept ) / slope )
    # return line points
    return np.array( [ x1, y1, x2, y2 ])

def display_lines(image, lines):
    # building black image
    line_image = np.zeros_like(image)
    # loop in the cordinates of each lines
    # line format ( x1, y1, x2, y2  )
    if lines is not None:
        for line in lines :
            x1, y1, x2, y2 = line.reshape(4)
            print(np.array( [ x1, y1, x2, y2 ]))
            # drow on the black image the lines
            # parameters ( the image that we want to drow on, the first line point
            #              the scond point, the color of the line, thickness of the line )
            try:
                cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 4 )
            except:
                continue
    return line_image

image = cv2.imread('image.png')
steering_angle = compute_steering_angle(image)
print("Steering Angle:",steering_angle)
slope = math.tan(steering_angle * math.pi / 180)
print("Slope:",slope)
line = make_coordinates(image, slope)
lane_image = np.copy(image)
# making the line image
line_image = display_lines( lane_image,  np.array([line]) )

# combine the orginal image with line line image
# parameters : (first image, weight1, second image, weight2, gamma )
combo_image = cv2.addWeighted(line_image, 0.99, lane_image, 1, 1 )


# display the image
cv2.imshow( 'sfs', line_image)
cv2.waitKey(0)


print(steering_angle)


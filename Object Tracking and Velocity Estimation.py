import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans as km
import cv2

# import warnings filter
from warnings import simplefilter
# ignore all "future warnings"
simplefilter(action='ignore', category=FutureWarning)


# ============================ calculating the pixel to feet conversion ratio ============================================

RGB = cv2.cvtColor(cv2.imread('im30.png'), cv2.COLOR_BGR2RGB) # when imported, this object is in BRG order, we need RGB

plt.figure(2)
plt.imshow(RGB)
plt.scatter([83, 305], [89, 89], marker='v', c='r')
plt.title('Click Below The Markers to Calibrate pixel/feet Conversion')
plt.show()
clicks = plt.ginput(2, show_clicks=True) # click on two points on the tape measure that are a known distance apart in feet
# feet = float(input("\nHow many feet apart  were your clicks in the image? (Input an integer, 1 or 2): "))
feet = 1
pixels2feet = feet / np.sqrt( (clicks[1][0] - clicks[0][0])**2 + (clicks[1][1] - clicks[0][1])**2)
print("The feet/pixels ratio is as follows: ", pixels2feet)

# ============================ processing the image to show how the distance calculations are found for each ball

processed = 255*np.uint8(np.logical_and(RGB[:,:,0]>130, RGB[:,:,1]<115))
# we need to create a logical array for the pixels that match these RBG value conditions
# the pink color value for the ball is found within above a Red value of 130 and below a green value of 115. 
# at the same time we are scaling that to 0's and 255's. that way it will be black and white
processed_blurred = cv2.medianBlur(processed,15)
# we need to blur it now, so we can get rid of the noise outside of the wiffle balls.
# this also fills in any holes

xVals, yVals = np.where(processed_blurred == 255) # turn the white values into x and y values
points = np.transpose(np.array([xVals, yVals])) # turn that into point pairs for the clustering algorithm
centers = km(n_clusters=4).fit(points).cluster_centers_ # this algorithm will find the centroids of 4 clusters

distances = np.zeros([4,4], dtype=float) # initialize a matrix of zeros for the distances between each respective wiffle ball
for i in np.arange(0, 4):
    for j in np.arange(0, 4):
        distances[j][i] = pixels2feet * np.sqrt( (centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2 )
print("\nThe following matrix shows the distance matrix for the balls in im30.mat. The measured distances are in feet.\n", distances)


# ============================ this section shows how the ball is tracked through the video object
# this section iterates through each frame of the video and plots the centroid of each ball to track their motion
# notice the camera's motion by the trail each "stationary" ball creates

vid = cv2.VideoCapture('wiffleBalls.mov')
thereAreFrames, BGR = vid.read() 
# this creates an image object and also creates the boolean that tells us if there is a frame after the current frame

plt.figure(3)
plt.title("Tracking the Wiffle Ball Motion")
while thereAreFrames:
    
    # each item in this while loop is differentiated from part 2 by the tagline _fromVid
    
    RGB_fromVid = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB) # when imported, this object is in BRG order, we need RGB for sanity's sake
    
    processed_fromVid = 255*np.uint8(np.logical_and(RGB_fromVid[:,:,0]>130, RGB_fromVid[:,:,1]<115)) 
    # we need to create a logical array for the pixels that match these RBG value conditions
    # at the same time we are scaling that to 0's and 255's. that way it will be black and white
    
    processed_fromVid_blurred = cv2.medianBlur(processed_fromVid,15) # blur to remove any small impurities in the image
    xVals_fromVid, yVals_fromVid = np.where(processed_fromVid_blurred == 255) # turn the balls into x and y values
    points_fromVid = np.transpose(np.array([xVals_fromVid, yVals_fromVid])) # turn that into point pairs
    centers_fromVid = km(n_clusters=4).fit(points_fromVid).cluster_centers_ # get the centers of each ball
    
    plt.imshow(processed_fromVid_blurred, cmap='gray') # show the processed image
    plt.scatter(centers_fromVid[:, 1], centers_fromVid[:, 0], c='r', s=2) # show the centers of each ball

    thereAreFrames, BGR = vid.read() # update the bool that shows there are still frames to read AND get the next frame
    
vid.release()
plt.show() 

# ============================ Estimating the ball speed

vid = cv2.VideoCapture('wiffleBalls.mov')
fps = vid.get(cv2.CAP_PROP_FPS) # we will use this to find the ball's speed
thereAreFrames, BGR = vid.read() 
# this creates an image object and also creates the boolean that tells us if there is a frame after the current frame
current_distances = np.zeros([4,4], dtype=float) # initialize a matrix of zeros
previous_distances = np.zeros([4,4], dtype=float) # initialize a matrix of zeros
time = np.zeros(int(vid.get(cv2.CAP_PROP_FRAME_COUNT)));
ball_velocity = np.zeros(int(vid.get(cv2.CAP_PROP_FRAME_COUNT)));

k = 0 # counter variable

while thereAreFrames:
    # each item in this while loop is differentiated from part 1 by the tagline _fromVid
    
    RGB_fromVid = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB) # when imported, this object is in BRG order, we need RGB
    
    processed_fromVid = 255*np.uint8(np.logical_and(RGB_fromVid[:,:,0]>130, RGB_fromVid[:,:,1]<115)) 
    # we need to create a logical array for the pixels that match these RBG value conditions
    # at the same time we are scaling that to 0's and 255's. that way it will be black and white
    
    processed_fromVid_blurred = cv2.medianBlur(processed_fromVid, 15) # blur to remove any small impurities in the image
    xVals_fromVid, yVals_fromVid = np.where(processed_fromVid_blurred == 255) # turn the balls into x and y values
    points_fromVid = np.transpose(np.array([xVals_fromVid, yVals_fromVid])) # turn that into point pairs
    centers_fromVid = km(n_clusters=4).fit(points_fromVid).cluster_centers_ # get the centers of each ball
    
    if(k==0):
        previous_centers = centers_fromVid
    if(k != 0):
        for i in range(4):
            for j in range(4):
                current_distances[j][i] = pixels2feet * np.sqrt((centers_fromVid[i][0] - previous_centers[j][0])**2 + (centers_fromVid[i][1] - previous_centers[j][1])**2)
    minDistances = np.zeros(4)
    for i in range(4):
        minDistances[i] = min(current_distances[:][i])
    if(k>0):
        ball_velocity[k] = max(minDistances) / (fps**-1)
        time[k] = time[k-1] + fps**-1
    k += 1
    previous_centers = centers_fromVid
    thereAreFrames, BGR = vid.read() # update the bool that shows there are still frames to read AND get the next frame
vid.release()
fig, ax = plt.subplots() 

ax.plot(time, ball_velocity)
ax.set(xlabel = 'Time (s)', ylabel = 'Velocity (ft/s)', title = 'Wiffle Speed Over Time')
plt.show()

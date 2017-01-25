#!/usr/bin/env python
import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt

# initialize the list of reference points and its respective counter
refPt = []	        # temporarily stores the coordinates of the vertices of each polygon drawn by the user 
cnt_refPt = 0 	        # vertices counter, maximum value = 4
creating_spot = False   # flag indicating the user is creating a new parking spot  
image_backup = 0

# stores the coordinate values of every click detected, 
def get_polygon_coordinates(event, x, y, flags, param):
    global refPt, cnt_refPt, image_backup, creating_spot, image
    
    if cnt_refPt < 4:
        if event == cv2.EVENT_LBUTTONUP:                                # check to see if the left mouse button was released
            if cnt_refPt == 0:                                          # create image back up before drawing anything to restore when user cancels operation
                image_backup = image.copy()
                creating_spot = True
            refPt.append((x, y))                                        # record the(x, y) coordinates
            cv2.circle(image,refPt[cnt_refPt], 2, (0,0,255), -1)        # draw a red point in every coordinate clicked by the user
            cv2.imshow("image", image)
            cnt_refPt = cnt_refPt + 1                                   # keep control of the amount of vertices

samples_a = []
samples_b = []
labels = []

def extract_roi():
    global refPt, cnt_refPt, image_backup, creating_spot, image, samples_a, samples_b, labels
    
    roi_corners = np.array([refPt[0],refPt[1],refPt[2],refPt[3]], np.int32)     # transform roi corners coordinates to cv2.polylines appropiate format
    roi_corners = roi_corners.reshape((-1,1,2))
    cv2.polylines(image,[roi_corners],True,(255,0,0))                           # draw red polygon around the parking spot
    cv2.imshow("image", image)
    print("Press 's' to save parking spot, press 'c' to cancel")        # ask user if he wants to store the parking spot
    key = cv2.waitKey(0)
    
    if key == ord("s"):                                                 # if the user wants to save it
        print("Saving parking spot...")

        print("Press 'e' if parking spot is empty, press 'o' if parking spor is occupied") 
        key = cv2.waitKey(0)

        if key == ord("e"):
            labels.append(0)
        elif key == ord("o"):
            labels.append(1)
        
        cv2.polylines(image,[roi_corners],True,(0,0,255))               # draw blue polygon around the parking spot
        cv2.imshow("image", image)
        
        min_x = min(refPt[0][0],refPt[1][0],refPt[2][0],refPt[3][0])    # get roi by extracting the smallest rectangle in which the polygon fits
        min_y = min(refPt[0][1],refPt[1][1],refPt[2][1],refPt[3][1])
        max_x = max(refPt[0][0],refPt[1][0],refPt[2][0],refPt[3][0])
        max_y = max(refPt[0][1],refPt[1][1],refPt[2][1],refPt[3][1])
        roi = clone[min_y:max_y, min_x:max_x]
        
        new_refPt = [(refPt[0][0]-min_x,refPt[0][1]-min_y),(refPt[1][0]-min_x,refPt[1][1]-min_y), # get new reference points (for the extraced roi)
                     (refPt[2][0]-min_x,refPt[2][1]-min_y),(refPt[3][0]-min_x,refPt[3][1]-min_y)]
        
        mask = np.zeros(roi.shape, dtype=np.uint8)                      # create mask for extracted roi (mask for visualization proposes)
        channel_count = roi.shape[2] 
        ignore_mask_color = (255,)*channel_count
        roi_corners = np.array([[new_refPt[0],new_refPt[1],new_refPt[2],new_refPt[3]]], np.int32)   # replace roi corners coordinates with the new reference points
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)                                          # and give them appropiate format for cv2.fillPolly  

        mask_1D = np.zeros(roi.shape[:2], dtype=np.uint8)               # create mask for extracted roi (mask to ignore area outside of hte polygon when
        ignore_mask_color = 255                                         # 
        cv2.fillPoly(mask_1D, roi_corners, ignore_mask_color)  
        Lab_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)

        hist_a = cv2.calcHist([Lab_roi],[1],mask_1D,[32],[0,255])
        hist_b = cv2.calcHist([Lab_roi],[2],mask_1D,[32],[0,255])
        
        plt.plot(hist_a,color = 'r')
        plt.plot(hist_b,color = 'b')
        plt.xlim([0,32])
        plt.show()
        color = ('r','b')                                               # ploting histograms

        hist_a = hist_a.flatten()
        hist_b = hist_b.flatten()

        samples_a.append(hist_a)
        samples_b.append(hist_b)
       
        masked_roi = cv2.bitwise_and(roi, mask)                         # apply the mask
        
        creating_spot = False
        cnt_refPt = 0                                                   # reinitialize reference points
        refPt = []
        return masked_roi
        
    elif key == ord("c"):                                               # if the user wants to cancel the operation
        print("Cancel...")
        image = image_backup.copy()                                     # restore image with backup
        cv2.imshow("image", image)
        creating_spot = False
        cnt_refPt = 0                                                   # reinitialize reference points
        refPt = []
        return [0]
            


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
#Lab_clone = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2LAB)
cv2.namedWindow("image")
cv2.setMouseCallback("image", get_polygon_coordinates)

# keep looping until the 'escape' key is pressed
while True:
    cv2.imshow("image", image)              # display the image and wait for a keypress
    key = cv2.waitKey(1) & 0xFF
        
    if key == ord("c") and creating_spot == True: # cancel 
        print("Cancel...")
        image = image_backup.copy()         # restoring image with backup
        cv2.imshow("image", image)
        creating_spot = False
        cnt_refPt = 0
        refPt = []
        
    # if there are 4 reference points, draw polygon then crop the region of interest
    # from the image and display it
    elif cnt_refPt == 4:
        roi = extract_roi()
        if len(roi) != 1:
            cv2.imshow("roi", roi)
            #lab_histogram(roi)
            
    # if the 'escape' key is pressed, break from the loop
    elif key == 27:
        break

print np.float32(labels)
print np.float32(samples_a)
print np.float32(samples_b)
# close all open windows
cv2.destroyAllWindows()

#!/usr/bin/env python
import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt

def menu():
    print '''
Training set generation of empty and occupied parking spots in a parking lot

Keys:
ESC     -   exit
n       -   create new region of ROI
c       -   cancel operation
s       -   save new ROI
    for labeling:
    e       -   to label as empty spot
    o       -   to label as occupied spot

'''    

'''
When the user has selected to create a new ROI, everytime she/he clicks in the image, a red dot is drawn in the clicked pixel and its coordinates are stored
until 4 clicks have been made
'''

font = cv2.FONT_HERSHEY_SIMPLEX

# initialize the list of reference points and its respective counter
refPt = []	        # temporarily stores the coordinates of the vertices of each polygon drawn by the user 
cnt_refPt = 0 	        # vertices counter, maximum value = 4
creating_spot = False   # flag indicating the user is creating a new parking spot  
image_backup = 0        # stores image backup

# stores the coordinate values of every detected click when creating a new parking spot
def get_polygon_coordinates(event, x, y, flags, param):
    global refPt, cnt_refPt, image_backup, creating_spot, image
    
    if cnt_refPt < 4 and creating_spot == True:                         # it stops storing coordinates when the 4 vertices have been drawn
        if event == cv2.EVENT_LBUTTONUP:                                # check to see if the left mouse button was released
            if cnt_refPt == 0:                                          # create image back up before drawing anything to restore when user cancels operation
                image_backup = image.copy()
                                                    
            refPt.append((x, y))                                        # record the(x, y) coordinates
            cv2.circle(image,refPt[cnt_refPt], 2, (0,0,255), -1)        # draw a red point in every coordinate clicked by the user
            cv2.imshow("image", image)
            cnt_refPt = cnt_refPt + 1                                   # keep control of the amount of vertices


'''
When the four vertices of the quadrilateral have been clicked, a polygon is drawn to show the user the selected region of interest. The user is then asked whether
he/she would like to save the current ROI. If the user chooses to save the ROI, then he is asked to choose if it is an empty or an occupied parking spot, a label
is created depending on its answer. The ROI goes later into a pre-processing stage, afterwards its features are extracted and stored.

If the user chooses not to save the ROI, all changes are discarded and the program returns to its initial state.
'''
# initialize lists that store features and labels
samples_a = []          # channel a histograms
samples_b = []          # channel b histograms
samples_H = []          # channel H histograms
samples_S = []          # channel S histograms
samples_HS = []
samples_f = []          # total keypoints detected by fast algorithm
labels = []             # labels selected by the user (0 = empty spot, 1 = occupied spot)

def label_and_extract_features():
    global refPt, cnt_refPt, image_backup, creating_spot, image, samples_a, samples_b, samples_H, samples_S, samples_HS, labels

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # A blue polygon is drawn to contour the selected ROI, the user is asked to save the ROI or cancel the operation                                            #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    roi_corners = np.array([refPt[0],refPt[1],refPt[2],refPt[3]], np.int32)     # transform roi corners coordinates to cv2.polylines appropiate format
    roi_corners = roi_corners.reshape((-1,1,2))
    cv2.polylines(image,[roi_corners],True,(255,0,0))                           # draw blue polygon around the parking spot
    cv2.imshow("image", image)
    print("Press 's' to save parking spot, press 'c' to cancel")                # ask user if she/he wants to store the parking spot
    key = cv2.waitKey(0)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # If the user chooses to store the ROI                                                                                                                      #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#    
    if key == ord("s"):                                                         
        print("Saving parking spot...")

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # then she/he is asked to manually select if the parking spot is empty or occupied, the labels are created with the selected answer                         #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
        while (key!=ord("e") and key!=ord("o")):
            print("Press 'e' if parking spot is empty, press 'o' if parking spot is occupied") 
            key = cv2.waitKey(0)

            if key == ord("e"):
                labels.append(0)
            elif key == ord("o"):
                labels.append(1)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # the blue polygon is replaced by a red one to indicate that the spot is being stored                                                                       #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#        
        cv2.polylines(image,[roi_corners],True,(0,0,255))                       # draw red polygon around the parking spot
        cv2.imshow("image", image)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # afterwards, the selected ROI is pre-processed                                                                                                             #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#        
        min_x = min(refPt[0][0],refPt[1][0],refPt[2][0],refPt[3][0])            # first, the coordinates of the vertices for the smallest bounding rectangle 
        min_y = min(refPt[0][1],refPt[1][1],refPt[2][1],refPt[3][1])            # in which the drawn polygon fits are computed
        max_x = max(refPt[0][0],refPt[1][0],refPt[2][0],refPt[3][0])
        max_y = max(refPt[0][1],refPt[1][1],refPt[2][1],refPt[3][1])

        cv2.putText(image,str(len(samples_f)),(min_x+5,max_y-5), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
        
        roi = clone[min_y:max_y, min_x:max_x]                                   # the rectangle is used to produce a new image containing solely the ROI

                                                                                # the ROI will be resized in order to obtain coherent features results
        x_size = 70                                                             # x_size and y_size store the new width and height values                                                     
        y_size = 70                                                             # due to the different possible angles that a parking spot can take
                                                                                # it is better to set these values to obtain a square

        scale_x = float(roi.shape[1])/float(x_size)                             # the scale of both axes is stored, it will be used to compute the new 
        scale_y = float(roi.shape[0])/float(y_size)                             # reference points

        roi = cv2.resize(roi,(x_size, y_size), interpolation = cv2.INTER_AREA)  # the roi is resized
         
        new_refPt = [((refPt[0][0]-min_x)/scale_x,(refPt[0][1]-min_y)/scale_y), # and its new reference points are computed
                     ((refPt[1][0]-min_x)/scale_x,(refPt[1][1]-min_y)/scale_y), 
                     ((refPt[2][0]-min_x)/scale_x,(refPt[2][1]-min_y)/scale_y),
                     ((refPt[3][0]-min_x)/scale_x,(refPt[3][1]-min_y)/scale_y)]

        #cv2.imshow("roi",roi)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # the pre-processing continues, masks are needed for extracting both, Lab histogram and fast detector features                                              #
    # to smooth the image, a gaussian filter is also applied before extracting the fast corner detector features                                                #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#    
        mask = np.zeros(roi.shape, dtype=np.uint8)                              # a 3-channel binary mask is created
        channel_count = roi.shape[2]                                            
        ignore_mask_color = (255,)*channel_count
        roi_corners = np.array([[new_refPt[0],new_refPt[1],                     # all pixels in the image which are not part of the polygon have a value of '0' 
                      new_refPt[2],new_refPt[3]]], np.int32)                    # while all pixels inside the polygon have a value of '255'
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)                    

        gaussian_roi = cv2.GaussianBlur(roi,(5,5),0)                            # apply a gaussian filter to reduce noise
        masked_roi = cv2.bitwise_and(gaussian_roi, mask)                        # apply the mask to implement fast algorithm

        cv2.imshow("masked_roi",masked_roi)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # Lab histogram features are extracted, the calcHist function already contains an option to ignore all the pixels that are not contained in a desired region#
    # a mask has to be passed in one of the parameters, in order to do this, we passed only one of the channels of the mask created in the previous section     #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#    
        Lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)                          # converting image from BGR space to Lab color space 

        hist_a = cv2.calcHist([Lab_roi],[1],mask[:,:,2],[32],[0,255])           # computing histograms for channels a and b, L is left out since it contains
        hist_b = cv2.calcHist([Lab_roi],[2],mask[:,:,2],[32],[0,255])           # information about the luminance
        
        hist_a = hist_a.flatten()                                               # histograms are converted to its appropiate format 
        hist_b = hist_b.flatten()

        samples_a.append(hist_a)                                                # and then stored in their respective sample arays
        samples_b.append(hist_b)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # Lab histogram features are extracted, the calcHist function already contains an option to ignore all the pixels that are not contained in a desired region#
    # a mask has to be passed in one of the parameters, in order to do this, we passed only one of the channels of the mask created in the previous section     #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#    
        HSV_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)                          # converting image from BGR space to Lab color space 

        hist_H = cv2.calcHist([HSV_roi],[0],mask[:,:,2],[32],[0,180])           # computing histograms for channels a and b, L is left out since it contains
        hist_S = cv2.calcHist([HSV_roi],[1],mask[:,:,2],[32],[0,256])           # information about the luminance
        
        hist_H = hist_H.flatten()                                               # histograms are converted to its appropiate format 
        hist_S = hist_S.flatten()
        
        
        samples_H.append(hist_H)                                                # and then stored in their respective sample arays
        samples_S.append(hist_S)

        temp_hist = np.append(hist_S, hist_H) 
        samples_HS.append(temp_hist)
        plt.plot(hist_a,color = 'b')
        plt.xlim([0,32])
        plt.show()
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # fast corner detector features are extracted from the filtered and masked ROI                                                                              #                                                                                              #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#          
        fast = cv2.FastFeatureDetector_create()                                 # initialize fast detector
        kp = fast.detect(masked_roi, None)                                      # implement it 	

        temp_img = cv2.drawKeypoints(gaussian_roi, kp, None, color=(0,255,0))   #storing image just for testing proposes
        cv2.imwrite('fast_imgPI.jpg',temp_img)
        
        samples_f.append(len(kp))                                               # store its total number of keypoints into its respective sample array                                    
        
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # if the user chooses to cancel the operation, the polygon and vertices are deleted and the program returns to its previous state                           #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#          
    elif key == ord("c"):                                               
        print("Cancel...")
        image = image_backup.copy()                                             # restore image with backup
        cv2.imshow("image", image)

    else:
        return
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # the flag, counter and reference points are reinitialized in order to start a new storing/extracting operation                                             #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    creating_spot = False
    cnt_refPt = 0                                                               # reinitialize reference points
    refPt = []
    
            

if __name__ == "__main__":
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    # load the image and setup the mouse callback function
    image = cv2.imread(args["image"])
    clone = image.copy()
    image_backup = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_polygon_coordinates)

    # print menu
    menu()
    
    # keep looping until the 'escape' key is pressed
    while True:
        cv2.imshow("image", image)                          # display the image and wait for a keypress

        key = cv2.waitKey(1) & 0xFF

    # if the user chooses to create a new ROI, label it and extract its features                                                 
        if key == ord("n") and creating_spot == False:
            print("Creating new ROI...")
            creating_spot = True
    # if the user chooses to cancel the operation, the vertices are deleted and the program returns to its previous state                                                 
        if key == ord("c") and creating_spot == True:  
            print("Cancel...")
            image = image_backup.copy()                     # restoring image with backup
            cv2.imshow("image", image)
            creating_spot = False
            cnt_refPt = 0
            refPt = []
            
    # if 4 reference points have been selected, ask user to save spot region or cancel operation                                                                
    # if save is selected, extract features and label spot                                                                                                      
        elif cnt_refPt == 4:
            label_and_extract_features()

    # if the 'escape' key is pressed, break from the loop                                                                                                                       
        elif key == 27:
            break

    # close all open windows
    cv2.destroyAllWindows()
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # store all samples and labels into a npz file                                                                                                              #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#           
    outfile = raw_input("Enter a filename: ")
    outimage = outfile + ".jpg"
    outfile = outfile + ".npz"
    #print samples_S
    #print samples_HS
    np.savez(outfile, labels=labels, samples_a=samples_a, samples_b=samples_b, samples_H=samples_H, samples_S=samples_S, samples_f=samples_f, samples_HS = samples_HS)
    cv2.imwrite(outimage, image)




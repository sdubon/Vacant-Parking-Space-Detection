#!/usr/bin/env python
import numpy as np
import cv2
import argparse

def menu():
    print '''
Definition of parking lot spots, this program has to be run
when the system is installed. The generated file should be
placed in the folder where 'parking_spot_classification.py'
is located.

Keys:
ESC     -   exit
n       -   create new region of ROI
c       -   cancel operation
s       -   save new ROI

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
he/she would like to save the current ROI. If the user chooses to save the ROI, its coordinates are stored in roi_coordinates
'''
roi_coordinates = []
def store_coordinates():
    global refPt, cnt_refPt, image_backup, creating_spot, image

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
    # the blue polygon is replaced by a red one to indicate that the spot is being stored                                                                       #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#        
        cv2.polylines(image,[roi_corners],True,(0,0,255))                       # draw red polygon around the parking spot
        cv2.imshow("image", image)

        roi_coordinates.append(refPt)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # show ROI to the user                                                                                                                                      #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#        
        min_x = min(refPt[0][0],refPt[1][0],refPt[2][0],refPt[3][0])            # first, the coordinates of the vertices for the smallest bounding rectangle 
        min_y = min(refPt[0][1],refPt[1][1],refPt[2][1],refPt[3][1])            # in which the drawn polygon fits are computed
        max_x = max(refPt[0][0],refPt[1][0],refPt[2][0],refPt[3][0])
        max_y = max(refPt[0][1],refPt[1][1],refPt[2][1],refPt[3][1])

        cv2.putText(image,str(len(roi_coordinates)),(refPt[1][0]+5,refPt[1][1]-5), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
        
        roi = clone[min_y:max_y, min_x:max_x]                                   # the rectangle is used to produce a new image containing solely the ROI

        cv2.imshow("roi",roi)
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
    # if save is selected, the coordinates are stored                                                                                                      
        elif cnt_refPt == 4:
            store_coordinates()

    # if the 'escape' key is pressed, break from the loop                                                                                                                       
        elif key == 27:
            break

    # close all open windows
    cv2.destroyAllWindows()
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # store all samples and labels into a npz file                                                                                                              #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#           
    outfile = 'coordinates.npz'
    np.savez(outfile, roi_coordinates = roi_coordinates)
    cv2.imwrite('coordinates.jpg', image)



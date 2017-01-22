# import the necessary packages
import numpy as np
import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
spots_vertices = []
masked_roi = []
cropping = False
i = 0 #vertices counter
j = 0 #spots counter

def store_parking_spot(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, i, image_backup
      
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		#refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:

                if i == 0:
                        # create image back up before drawing anything
                        print("creating image backup")
                        image_backup = image.copy()
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw red point where click was made
		cv2.circle(image,refPt[i], 2, (0,0,255), -1)
		cv2.imshow("image", image)
		i = i + 1
		#print (len(refPt))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", store_parking_spot)

# keep looping until the 'escape' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == 27:
		break

        # if there are 4 reference points, draw polygon then crop the region of interest
        # from the image and display it
        elif len(refPt) == 4:
                # create image back up before drawing anything
                # draw polygon
                pts = np.array([refPt[0],refPt[1],refPt[2],refPt[3]], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(image,[pts],True,(255,0,0))
                cv2.imshow("image", image)
                # ask user if he wants to store the parking spot
                print("Press 's' to save parking spot, press 'c' to cancel")
                key = cv2.waitKey(0)
                if key == ord("s"):
                        print("Saving parking spot...")
                        cv2.polylines(image,[pts],True,(0,0,255))
                        cv2.imshow("image", image)
                        spots_vertices.append([refPt[0],refPt[1],refPt[2],refPt[3]])
                        min_x = min(refPt[0][0],refPt[1][0],refPt[2][0],refPt[3][0])
                        min_y = min(refPt[0][1],refPt[1][1],refPt[2][1],refPt[3][1])
                        max_x = max(refPt[0][0],refPt[1][0],refPt[2][0],refPt[3][0])
                        max_y = max(refPt[0][1],refPt[1][1],refPt[2][1],refPt[3][1])
                        new_refPt = [(refPt[0][0]-min_x,refPt[0][1]-min_y),(refPt[1][0]-min_x,refPt[1][1]-min_y),(refPt[2][0]-min_x,refPt[2][1]-min_y),(refPt[3][0]-min_x,refPt[3][1]-min_y)]

                        # getting subimage
                        roi = clone[min_y:max_y, min_x:max_x]
                        
                        # mask defaulting to black for 3-channel and transparent for 4-channel
                        # (of course replace corners with yours)
                        mask = np.zeros(roi.shape, dtype=np.uint8)
                        roi_corners = np.array([[new_refPt[0],new_refPt[1],new_refPt[2],new_refPt[3]]], np.int32)
                        
                        # fill the ROI so it doesn't get wiped out when the mask is applied
                        channel_count = roi.shape[2]  # i.e. 3 or 4 depending on your image
                        print(channel_count)
                        ignore_mask_color = (255,)*channel_count
                        
                        print(ignore_mask_color)
                        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                        # apply the mask
                        masked_roi.append(cv2.bitwise_and(roi, mask))

                        #cv2.imshow("ROI", roi)
                        cv2.imshow("masked roi", masked_roi[j])
                        j = j + 1
                        #cv2.waitKey(0)
                elif key == ord("c"):
                        # restoring image with backup
                        print("Cancel...")
                        image = image_backup.copy()
                        cv2.imshow("image", image)
                i = 0
                refPt = []
                

print(spots_vertices)
print(new_refPt)
print(len(masked_roi))
# close all open windows
cv2.destroyAllWindows()

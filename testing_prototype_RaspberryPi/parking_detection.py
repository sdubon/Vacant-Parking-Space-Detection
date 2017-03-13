# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import time

# built-in modules
import os
import sys
import MySQLdb as msql

####### SQL configuration ########
config = {
    'user': 'root',
    'passwd': 'A&1bC&2d',
    'host':'192.168.43.241',
    'db':'parking',
    }
	
def exe_query(db, dataa,index):
	cur = db.cursor()
	result=cur.execute("UPDATE `FINAL_parking` SET state = "+dataa+" WHERE id = "+index+"")
	db.commit()
	cur.close()

####### Machine Learning functions #####
class StatModel(object):
    def load(self, fn):
        self.model.load(fn) 
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5, kernel = cv2.ml.SVM_RBF):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(kernel)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


refPt = []
masks = []
min_x = []
min_y = []
max_x = []
max_y = []
x_size = 70
y_size = 70
def initialization():
    global refPt, masks, min_x, min_y, max_x, max_y, x_size, y_size 

    for i in range (len(refPt)):
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # getting relative coordinates from original coordinates                                                                                                    #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#        
        min_x.append(min(refPt[i][0][0],refPt[i][1][0],refPt[i][2][0],refPt[i][3][0]))# first, the coordinates of the vertices for the smallest bounding rectangle 
        min_y.append(min(refPt[i][0][1],refPt[i][1][1],refPt[i][2][1],refPt[i][3][1]))# in which the drawn polygon fits are computed
        max_x.append(max(refPt[i][0][0],refPt[i][1][0],refPt[i][2][0],refPt[i][3][0]))
        max_y.append(max(refPt[i][0][1],refPt[i][1][1],refPt[i][2][1],refPt[i][3][1]))
       
        scale_x = float(max_x[i]-min_x[i])/float(x_size)                             # the scale of both axes is stored, it will be used to compute the new 
        scale_y = float(max_y[i]-min_y[i])/float(y_size)                             # reference points
     
        new_refPt = [((refPt[i][0][0]-min_x[i])/scale_x,(refPt[i][0][1]-min_y[i])/scale_y), # and its new reference points are computed
                     ((refPt[i][1][0]-min_x[i])/scale_x,(refPt[i][1][1]-min_y[i])/scale_y), 
                     ((refPt[i][2][0]-min_x[i])/scale_x,(refPt[i][2][1]-min_y[i])/scale_y),
                     ((refPt[i][3][0]-min_x[i])/scale_x,(refPt[i][3][1]-min_y[i])/scale_y)]
        
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # initializing masks for every parking spot                                                                                                                 #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#    
        mask = np.zeros((y_size,x_size,3), dtype=np.uint8)                   # a 3-channel binary mask is created                                            
        ignore_mask_color = (255, 255, 255)
        roi_corners = np.array([[new_refPt[0],new_refPt[1],               # all pixels in the image which are not part of the polygon have a value of '0' 
                      new_refPt[2],new_refPt[3]]], np.int32)              # while all pixels inside the polygon have a value of '255'
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)                    

        masks.append(mask)
 
def update_samples(img):
    global refPt, masks, min_x, min_y, max_x, max_y, x_size, y_size
    samples_f = []
    samples_HS = []
   
    for i in range (len(refPt)): 
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # every ROI is pre-processed                                                                                                             #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#        
        roi = img[min_y[i]:max_y[i], min_x[i]:max_x[i]]                         # the rectangle is used to produce a new image containing solely the ROI
        roi = cv2.resize(roi,(x_size, y_size), interpolation = cv2.INTER_AREA)  # the roi is resized

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # the pre-processing continues, masks are needed for extracting both, Lab histogram and fast detector features                                              #
    # to smooth the image, a gaussian filter is also applied before extracting the fast corner detector features                                                #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#    
        gaussian_roi = cv2.GaussianBlur(roi,(5,5),0)                            # apply a gaussian filter to reduce noise
        masked_roi = cv2.bitwise_and(gaussian_roi, masks[i])                    # apply the mask to implement fast algorithm
            
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # HSV histogram features are extracted, the calcHist function already contains an option to ignore all the pixels that are not contained in a desired region#
    # a mask has to be passed in one of the parameters, in order to do this, we passed only one of the channels of the mask created in the previous section     #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#    
        HSV_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)                          # converting image from BGR space to Lab color space 

        hist_H = cv2.calcHist([HSV_roi],[0],masks[i][:,:,2],[32],[0,255])           # computing histograms for channels a and b, L is left out since it contains
        hist_S = cv2.calcHist([HSV_roi],[1],masks[i][:,:,2],[32],[0,255])           # information about the luminance
        
        hist_H = hist_H.flatten()                                               # histograms are converted to its appropiate format 
        hist_S = hist_S.flatten()
        
        temp_hist = np.append(hist_S, hist_H) 
        samples_HS.append(temp_hist)
        
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # fast corner detector features are extracted from the filtered and masked ROI                                                                              #                                                                                              #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#          
        fast = cv2.FastFeatureDetector_create()                                 # initialize fast detector
        kp = fast.detect(masked_roi, None)                                      # implement it 	

        samples_f.append(len(kp))                                               # store its total number of keypoints into its respective sample array

    return np.float32(samples_f), np.float32(samples_HS)

def draw_rectangles(img, group):
    global refPt, masks, min_x, min_y, max_x, max_y, x_size, y_size 

    for i in range (len(refPt)):

        if group[1][i] == 1:
            color = (255,0,0)
        else:
            color = (0,255,0)
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
        # A blue polygon is drawn to contour the selected ROI, the user is asked to save the ROI or cancel the operation                                            #
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
        roi_corners = np.array([refPt[i][0],refPt[i][1],refPt[i][2],refPt[i][3]], np.int32)     # transform roi corners coordinates to cv2.polylines appropiate format
        roi_corners = roi_corners.reshape((-1,1,2))
        cv2.polylines(img,[roi_corners],True,color)                           # draw blue polygon around the parking spot

    cv2.imshow("image", img)

def update_server(db, group):
    global refPt
    for i in range (len(refPt)):
            if int(group[1][i][0]) == 1:
                    exe_query(db,'0',str(i+1))
            else:
                    exe_query(db,'1',str(i+1))
    
def main():
    global refPt

    #connect to server
    db = msql.connect(**config)
    
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
     
    # allow the camera to warmup
    time.sleep(0.1)
    
    # loading coordinates
    foo = np.load('coordinates.npz')
    refPt = foo['roi_coordinates']
    
    initialization()
        
    classifier_f = 'fast_svm.dat'
    if not os.path.exists(classifier_f):
        print('"%s" not found, run svm_training.py first' % classifier_f)
        return

    classifier_HS = 'channel_HS_svm.dat'
    if not os.path.exists(classifier_HS):
        print('"%s" not found, run svm_training.py first' % classifier_HS)
        return
    
    if True:
        model_f = cv2.ml.SVM_load(classifier_f)
        model_HS = cv2.ml.SVM_load(classifier_HS)
    else:
        model_f = cv2.ml.SVM_create()
        model_f.load_(classifier_f)
        model_HS = cv2.ml.SVM_create()
        model_HS.load_(classifier_HS)

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        samples_f, samples_HS = update_samples(image)

        result_f = model_f.predict(samples_f)
        result_HS = model_HS.predict(samples_HS)

        #showing image with rectangles in every place
        draw_rectangles(image, result_f)

        #updating parking info on server
        update_server(db, result_f)
        
        key = cv2.waitKey(4000) & 0xFF
 
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
                
    # close all open windows
    cv2.destroyAllWindows()
    
    # close server connection
    db.close()
    
if __name__ == '__main__':
    main()


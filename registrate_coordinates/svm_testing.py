import cv2
import numpy as np

# built-in modules
import os
import sys

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
    # loading coordinates
    foo = np.load('coordinates.npz')
    refPt = foo['roi_coordinates']

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

    samples_a = []
    samples_b = []
    samples_f = []

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
        ch = cv2.waitKey(0)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # Lab histogram features are extracted, the calcHist function already contains an option to ignore all the pixels that are not contained in a desired region#
    # a mask has to be passed in one of the parameters, in order to do this, we passed only one of the channels of the mask created in the previous section     #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#    
        Lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)                          # converting image from BGR space to Lab color space 

        hist_a = cv2.calcHist([Lab_roi],[1],masks[i][:,:,2],[32],[0,255])       # computing histograms for channels a and b, L is left out since it contains
        hist_b = cv2.calcHist([Lab_roi],[2],masks[i][:,:,2],[32],[0,255])       # information about the luminance
        
        hist_a = hist_a.flatten()                                               # histograms are converted to its appropiate format 
        hist_b = hist_b.flatten()

        samples_a.append(hist_a)                                                # and then stored in their respective sample arrays
        samples_b.append(hist_b)

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    # fast corner detector features are extracted from the filtered and masked ROI                                                                              #                                                                                              #
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------#          
        fast = cv2.FastFeatureDetector_create()                                 # initialize fast detector
        kp = fast.detect(masked_roi, None)                                      # implement it 	

        samples_f.append(len(kp))                                               # store its total number of keypoints into its respective sample array

    print "a", np.float32(samples_a)
    print "b", np.float32(samples_b)
    return np.float32(samples_a), np.float32(samples_b), np.float32(samples_f)
        
def main():

    initialization()
    
    classifier_a = 'channel_a_svm.dat'
    if not os.path.exists(classifier_a):
        print('"%s" not found, run svm_training.py first' % classifier_a)
        return
    
    classifier_b = 'channel_b_svm.dat'
    if not os.path.exists(classifier_b):
        print('"%s" not found, run svm_training.py first' % classifier_b)
        return
    
    classifier_f = 'fast_svm.dat'
    if not os.path.exists(classifier_f):
        print('"%s" not found, run svm_training.py first' % classifier_f)
        return
    
    if True:
        model_a = cv2.ml.SVM_load(classifier_a)
        model_b = cv2.ml.SVM_load(classifier_b)
        model_f = cv2.ml.SVM_load(classifier_f)
    else:
        model_a = cv2.ml.SVM_create()
        model_a.load_(classifier_a) 
        model_b = cv2.ml.SVM_create()
        model_b.load_(classifier_b)
        model_f = cv2.ml.SVM_create()
        model_f.load_(classifier_f)
        
    image = cv2.imread('image3.jpg')
    samples_a, samples_b, samples_f = update_samples(image)

    result_a = model_a.predict(samples_a)
    result_b = model_b.predict(samples_b)
    result_f = model_f.predict(samples_f)

    print result_a[1]
    print result_b[1]
    print result_f[1]
    
    ch = cv2.waitKey(0)

if __name__ == '__main__':
    main()


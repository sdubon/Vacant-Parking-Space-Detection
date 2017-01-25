import numpy as np
import cv2


# Creation de l'image
img = cv2.imread('image3.jpg')

# Creation de l'image filtree

gaussian_roi = cv2.GaussianBlur(img,(5,5),0)

# Initialisation du FAST
fast = cv2.FastFeatureDetector_create()

# trouver et dessiner les points d'interet
kp = fast.detect(gaussian_roi, None) 				#Donne le nombre de point d'interet
imgPI = cv2.drawKeypoints(gaussian_roi, kp, None, color=(0,255,0))	#Dessine les points d'interet

cv2.imwrite('fast_imgPI.jpg',imgPI)				# Enregistre l'image comportant les points d'interet

print "Nombre point test image filtre : ",len(kp)


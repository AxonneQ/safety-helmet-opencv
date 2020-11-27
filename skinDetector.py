# First of all we initialize an object using an image. From here the image is converted to an HSV Image
#
# findSkin() is a function that by calling 2 other functions, it returns the final mask containing the skin.
# First we call the color segmentation that will segment the HSV picture in a mask.
# Then the region segmentation, by using some morphological operation returns the final skin mask
#
# The color segmentation method returns all the pixels contained between a lower and upper range. These values were found by empirical experimentation and 
# are giving good results to almost all the pictures.
# 
# The region Segmentation first will erode the mask from the color segmentation in order to find the sure foreground. 
# Then the dilatation of the mask will give back what is the sure background. 
# The marker will be the sum of the two images so we can find the region that is unknown. 
# By using the cv2.watershed method, a full mask containing all the informations required is returned


import numpy as np
import cv2

class SkinDetector(object):
    # initialization an object using an image. From here the image is converted to an HSV Image
    def __init__(self, img):
        self.imgMask = []
        self.output = []
        self.image = img

        self.HSVImg = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.binaryMask = self.HSVImg
    
    #findSkin() is a function that by calling 2 other functions, it returns the final mask containing the skin.
    # First we call the color segmentation that will segment the HSV picture in a mask.
    # Then the region segmentation, by using some morphological operation returns the final skin mask
    def findSkin(self):
        self.colorSegmentation()
        self.regionSegmentation()

    # The color segmentation method returns all the pixels contained between a lower and upper range. These values were found by empirical experimentation and 
    # are giving good results to almost all the pictures.
    def colorSegmentation(self):
        lowerHSV = np.array([0, 40, 0], dtype="uint8")
        upperHSV = np.array([25, 210, 240], dtype="uint8")

        maskHSV = cv2.inRange(
            self.HSVImg, lowerHSV, upperHSV)

        self.binaryMask = maskHSV
        
    #The region Segmentation first will erode the mask from the color segmentation in order to find the sure foreground. 
    # Then the dilatation of the mask will give back what is the sure background. 
    # The marker will be the sum of the two images so we can find the region that is unknown. 
    # By using the cv2.watershed method, a full mask containing all the informations required is returned
    def regionSegmentation(self):

        fgImg = cv2.erode(
            # remove noise
            self.binaryMask, None, iterations=3)  

        dilatedMask = cv2.dilate(
            self.binaryMask, None, iterations=3)
        ret, bgImg = cv2.threshold(
            dilatedMask, 1, 128, cv2.THRESH_BINARY) 

        marker = cv2.add(fgImg, bgImg)

        # convert to uint32 for watershedding
        marker32 = np.int32(marker)

        cv2.watershed(self.image, marker32)

        # convert back to uint8 for processing
        m = cv2.convertScaleAbs(marker32)  

        ret, mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.imgMask = mask
        self.output = cv2.bitwise_and(self.image, self.image, mask=mask)

    def getMask(self):
        return self.imgMask

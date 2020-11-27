import numpy as np
import cv2

class SkinDetector(object):

    def __init__(self, img):
        self.imgMask = []
        self.output = []
        self.image = img

        self.HSVImg = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.binaryMask = self.HSVImg

    def findSkin(self):
        self.colorSegmentation()
        self.regionSegmentation()

    def colorSegmentation(self):
        lowerHSV = np.array([0, 40, 0], dtype="uint8")
        upperHSV = np.array([25, 210, 240], dtype="uint8")

        maskHSV = cv2.inRange(
            self.HSVImg, lowerHSV, upperHSV)

        self.binaryMask = maskHSV

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

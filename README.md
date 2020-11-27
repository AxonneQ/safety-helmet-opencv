# safety-helmet-opencv
4th year Image Processing project to identify whether person is wearing safety helmet on construction site or not

### Instructions:

Pre-requisites:
```
OpenCV
NumPy
cvui
easygui
matplotlib
```

1. Run `SHDetector.py` with Python 3.8
2. Click Load Image button in the UI
3. Click Safety Helmet Detection
4. Optionally Save the output to a file using Save Image button

Pseudocode:

preProcess():
* Convert to RGB, and remove the noise using `cv2.fastNlMeansDenoising`

getSkinMask():
* SkinDetector Class
* HSV color segmentation of skin values
* WaterShedding
* otsu thresholding
* return skin mask

getFaces():
* generate sorted contours from the skinMask
* discard any contour bounding rect that we deem too small by comparing its area to the largest area. If an area is ⅕ the size of the largest we discard it.
* We then combine all the clusters of rectangles that are touching each other.
* If a rectangle contains any children then discard them too
* If by this point we have no values from prev functions then we assume our contours correctly picked out the correct faces and reset back to these values instead.
* A face helmet will always be on the upper body in the image, therefore we discard any contours that appear under a contour.
* We now reliably have a face detection but we need to double its height upwards in order to search for the helmet as well.

getHelmets(src_img, skinMask, faces):
* convert the original image to rgb
* get a region of image (ROI) using the inverted skinMask
* get rid of all values that are below 150, above 250 ro get rid of the background
* for each face region determine whether it includes a helmet using `processHelmet(face_image)` function
* if the function returns true, draw a green rectangle around the face else draw a red one

processHelmet(face_image):
* calculate the area of the face
* find range of predetermined HSV values of the helmets (specific colours)
* if area of the helmet takes more than 39.0 % then return true else false

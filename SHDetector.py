'''
    Safety Helmet Detector
    Image Processing Group Assignment 2020
    Authors: Igor Bolek, Vlad Medves, Iosif Balasca

    The objective of this assignment is to create a program to detect whether a people on construction site are
    wearing a safety helmet. The program should:
        1. Allow user to load an image.
        2. Detect people WITHOUT helmets. (highlighted by a red bounding box)
        3. Detect people WITH safety helmets on. (highlighted by a green bounding box)
        4. Allow user to save the processed image.
'''

import cv2
import numpy as np
import cvui
import easygui

class SDHException(Exception):
    def __init__(self, exception):
        self.exception = exception

def detect_safety(src_img):

    # Call Human Detection function
    detected_people = detect_people(src_img)

    # Call Helmet Detection function
    detected_helmets = detect_helmets(src_img, detected_people)

    # Call deciding logic function

    return detected_helmets

def detect_people(src_img):
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    body_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_upperbody.xml')

    return body_cascade.detectMultiScale(gray, 1.03, 3)

def detect_helmets(src_img, bodies):
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    src_height, src_width = source_img.shape[:2]

    for(x, y, w, h) in bodies:
        roi = np.zeros((src_height, src_width), np.uint8)
        roi[y:y+h, x:x+w] = gray[y:y+h, x:x+w]

        thresh, mask = cv2.threshold(roi, thresh = 220, maxval = 255, type = cv2.THRESH_BINARY)
        new_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE ,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)))

        contours, _ = cv2.findContours(new_mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cx, cy, cw, ch = cv2.boundingRect(contours[0])
            cv2.rectangle(src_img, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
            src_img = cv2.rectangle(src_img, (x,y),(x+w,y+h),(255,0,0),2)

    return src_img

#################### Main driver program #######################

# Main UI Frame and Source Image variable
window_name = 'Safety Helmet Detector'

ui_width = 500
ui_height = 200

max_image_width = 480

toolbar_top_height = 100

# Image Containers
frame = np.zeros((ui_height, ui_width, 3), np.uint8)
source_img = np.array([])
detected_img = np.array([])

image_loaded = False
image_padding = 10

# Button messages
load_action_message = 'Please choose an image...'
load_action_message_color = 0xCECECE

detect_action_message = ''
detect_action_message_color = 0xCECECE

cvui.init(window_name)

# main program loop (window property check as while condition allows the close button to end the program)
while cv2.getWindowProperty(window_name, 0) >= 0:
    
    # If image is loaded adjust the UI size and display the image, else use default values
    if source_img.size != 0:
        src_height, src_width = source_img.shape[:2]

        new_height = src_height + toolbar_top_height + (image_padding * 2)
        new_width = src_width + (image_padding * 2)

        frame = np.zeros((new_height, new_width, 3), np.uint8)
        frame[:] = (49, 52, 49)
        
        if detected_img.size != 0:
            cvui.image(frame, image_padding, toolbar_top_height + image_padding, detected_img)
        else:
            cvui.image(frame, image_padding, toolbar_top_height + image_padding, source_img)
    else:
        frame = np.zeros((ui_height, ui_width, 3), np.uint8)
        frame[:] = (49, 52, 49)

    # Load Image Button. if clicked, easygui opens a dialog for opening images. Error checking included.
    b_load = cvui.button(frame, 10, 10, 'Load Image')
    if b_load:
        src_path = easygui.fileopenbox('Choose an image...', filetypes=[['*.jpg', '*.png', '*.bmp', 'Image Files'], '*'])
        if src_path != None:
            try:
                source_img = cv2.imread(src_path)

                if isinstance(source_img, type(None)):
                    source_img = np.array([])
                    raise SDHException('Wrong File Type')

                src_height, src_width = source_img.shape[:2]
                scale = max_image_width / src_width                
                
                source_img = cv2.resize(source_img, (int(src_width * scale), int(src_height * scale)), cv2.INTER_AREA)
                load_action_message = src_path
                load_action_message_color = 0xCECECE
                detect_action_message = ''
                detect_action_message_color = 0xCECECE
                detected_img = np.array([])

            except SDHException:
                load_action_message = 'Wrong file type. Please open an image file.'
                load_action_message_color = 0xFF0000

    # Adding text beside the button to display path or error message
    cvui.text(frame, 126, 18, load_action_message, 0.4 , load_action_message_color)

    # If the image was loaded successfully, 2 buttons appear 'Safety Detection' and 'Save Image'
    if source_img.size != 0:
        b_detect = cvui.button(frame, 10, 44, 'Safety Detection')
        if b_detect:
            detected_img = detect_safety(source_img)
            detect_action_message = 'Done!'
            detect_action_message_color = 0x00FF00
            
        cvui.text(frame, 167, 52, detect_action_message, 0.4, detect_action_message_color)

        b_save = cvui.button(frame, frame.shape[1] - 104 - image_padding, 44, 'Save Image')
        if b_save:
            save_path = easygui.filesavebox('Save Image..', default='detected.jpg', filetypes=['*.jpg'])
            if not isinstance(save_path, type(None)):
                if detected_img.size != 0:
                    cv2.imwrite(save_path, detected_img)
                else:
                    cv2.imwrite(save_path, source_img)

    # Show the output on screen
    cvui.imshow(window_name, frame)

    # Exit using ESC button
    if cv2.waitKey(20) == 27:
        break



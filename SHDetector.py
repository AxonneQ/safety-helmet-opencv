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
from processImage import process as processImage

class SDHException(Exception):
    def __init__(self, exception):
        self.exception = exception

def detect_safety(src_img):
    img = processImage(src_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

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
source_img_copy = np.array([])
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

                source_img_copy = source_img.copy()
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
    if source_img_copy.size != 0:
        b_detect = cvui.button(frame, 10, 44, 'Safety Detection')
        if b_detect:
            detected_img = detect_safety(source_img_copy)
            detect_action_message = 'Done!'
            detect_action_message_color = 0x00FF00

            det_h, det_w = source_img.shape[:2]
            scale = max_image_width / det_w                  
            detected_img = cv2.resize(detected_img, (int(det_w * scale), int(det_h * scale)), cv2.INTER_AREA)
            
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



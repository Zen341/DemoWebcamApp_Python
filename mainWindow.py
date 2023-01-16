from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime as dt
from pathlib import Path
import os

# Define a global Mat variable
frame = np.zeros((rows, columns, channels), dtype = "uint8")

# Define directory to save samples
projDir = os.path.dirname(os.path.abspath(__file__))
smplImgPath = projDir + "\\sample\\images\\"
Path(smplImgPath).mkdir(parents=True, exist_ok=True)

# Define a video capture object
vid = cv2.VideoCapture(0)

# Declare the width and height in variables
width, height = 800, 600

# Set the width and height
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Create a GUI app
app = Tk()

# Bind the app with Escape keyboard to
# quit app whenever pressed
app.bind('<Escape>', lambda e: app.quit())

# Create a label and display it on app
label_widget = Label(app)
label_widget.pack()

isSharpen = IntVar()
chbxImgSharp = Checkbutton(app, text='Image sharpening', variable=isSharpen, onvalue=1, offvalue=0, command=lambda: print("check changed: " + str(isSharpen.get()))).pack()


# Image sharpening
def imgSharp(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp


# Create a function to open camera and
# display it in the label_widget on app
def open_camera():
    # Capture the video frame by frame
    _, frame = vid.read()

    # Image enhance
    # Sharpening
    if isSharpen.get() == 1:
        frame = imgSharp(frame)

    # Convert image from one color space to other
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    # Capture the latest frame and transform to image
    captured_image = Image.fromarray(opencv_image)

    # Convert captured image to photoimage
    photo_image = ImageTk.PhotoImage(image=captured_image)

    # Displaying photoimage in the label
    label_widget.photo_image = photo_image

    # Configure image in the label
    label_widget.configure(image=photo_image)

    # Repeat the same process after every 10 seconds
    label_widget.after(10, open_camera)

# The app will open camera automatically
# # Create a button to open the camera in GUI app
# button1 = Button(app, text="Open Camera", command=open_camera)
# button1.pack()


# Create a function to save image
def save_image():
    # _, frame = vid.read()
    now = dt.now()
    filename = smplImgPath + 'image_' + now.strftime("%Y_%m_%d_%H_%M_%S") + '.jpg'
    cv2.imwrite(filename, frame)


# Create a button to save image
btnSave = Button(app, text="Capture", command=save_image)
btnSave.pack()

# Open camera after the app displayed for 1 second
app.after(1000, open_camera())

# Create an infinite loop for displaying app on screen (center screen)
app.eval('tk::PlaceWindow . center')
app.mainloop()

import os
from datetime import datetime as dt
from pathlib import Path
from tkinter import *

import cv2
import numpy as np
from PIL import Image, ImageTk
from wand.image import Image as WandImg

# region Initial Setup
# Define directory to save samples
projDir = os.path.dirname(os.path.abspath(__file__))
smplImgPath = projDir + "\\sample\\images\\"
Path(smplImgPath).mkdir(parents=True, exist_ok=True)

# Define a video capture object
vid = cv2.VideoCapture(0)

# Declare the width and height in variables
width, height = 640, 480

# Set the width and height
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Set values for radio buttons
rValue = {"No Effect": "0",
          "Mirror": "1",
          "Kaleidoscope": "2",
          "Swirl": "3",
          "Light Tunnel": "4"}


# endregion

# region Methods, Functions
# Image sharpening
def img_sharp(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp


# Image Mirror Effect
def img_mirror(image):
    # Actual width and height
    h = height
    w = width
    half = w // 2

    # this will be the first column
    left_part = image[:, :half]
    # [:,:half] means all the rows and
    # all the columns upto index half
    right_part = cv2.flip(left_part, 1)

    left_image = Image.fromarray(cv2.cvtColor(left_part, cv2.COLOR_BGR2RGB))

    right_image = Image.fromarray(cv2.cvtColor(right_part, cv2.COLOR_BGR2RGB))

    new_img = Image.new('RGB', (w, h))
    new_img.paste(left_image, (0, 0))
    new_img.paste(right_image, (half, 0))

    return np.asarray(new_img)[:, :, ::-1].copy()


# Image Kaleidoscope Effect
def img_kaleidoscope(img):
    # Crop image to square shape
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    w, h = pil_img.size  # Get dimensions

    new_width = 480
    new_height = 480

    left = (w - new_width) / 2
    top = (h - new_height) / 2
    right = (w + new_width) / 2
    bottom = (h + new_height) / 2

    # Crop the center of the image
    pil_img = pil_img.crop((left, top, right, bottom))
    img = np.asarray(pil_img)[:, :, ::-1].copy()

    # arguments
    invert = "yes"  # invert mask; yes or no
    rotate = 0  # rotate composite; 0, 90, 180, 270

    ht, wd = img.shape[:2]

    # transpose the image
    imgt = cv2.transpose(img)

    # create diagonal bi-tonal mask
    mask = np.zeros((ht, wd), dtype=np.uint8)
    points = np.array([[[0, 0], [wd, 0], [wd, ht]]])
    cv2.fillPoly(mask, points, 255)
    if invert == "yes":
        mask = cv2.bitwise_not(mask)

    # composite img and imgt using mask
    compA = cv2.bitwise_and(imgt, imgt, mask=mask)
    compB = cv2.bitwise_and(img, img, mask=255 - mask)
    comp = cv2.add(compA, compB)

    # rotate composite
    if rotate == 90:
        comp = cv2.rotate(comp, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        comp = cv2.rotate(comp, cv2.ROTATE_180)
    elif rotate == 270:
        comp = cv2.rotate(comp, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # mirror (flip) horizontally
    mirror = cv2.flip(comp, 1)

    # concatenate horizontally
    top = np.hstack((comp, mirror))

    # mirror (flip) vertically
    bottom = cv2.flip(top, 0)

    # concatenate vertically
    kaleidoscope_big = np.vstack((top, bottom))

    # resize
    kaleidoscope = cv2.resize(kaleidoscope_big, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    return kaleidoscope


# Image Swirl Effect
def img_swirl(image):
    # pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # pil_image.swirl(degree=360)
    # return np.asarray(pil_image)[:, :, ::-1].copy()
    wand_img = WandImg.from_array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    wand_img.swirl(degree=120)
    return np.asarray(wand_img)[:, :, ::-1].copy()


# Image Light Tunnel Effect
def img_light_tunnel(im):
    (h, w) = im.shape[:2]

    # some definitions
    center = np.array([w / 2, h / 2])
    radius = h / 2.5

    i, j = np.mgrid[0:h, 0:w]
    xymap = np.dstack([j, i]).astype(np.float32)  # "identity" map

    # coordinates relative to center
    coords = (xymap - center)
    # distance to center
    dist = np.linalg.norm(coords, axis=2)
    # touch only what's outside of the circle
    mask = (dist >= radius)
    # project onto circle (calculate unit vectors, move onto circle, then back to top-left origin)
    xymap[mask] = coords[mask] / dist[mask, None] * radius + center

    out = cv2.remap(im, map1=xymap, map2=None, interpolation=cv2.INTER_LINEAR)
    return out


# Create a function to open camera and
# display it in the label_widget on app
def open_camera():
    # Capture the video frame by frame
    _, frame = vid.read()

    # Image effect
    if effect.get() == "1":
        frame = img_mirror(frame)
    if effect.get() == "2":
        frame = img_kaleidoscope(frame)
    if effect.get() == "3":
        frame = img_swirl(frame)
    if effect.get() == "4":
        frame = img_light_tunnel(frame)

    # Image enhance
    # Sharpening
    if isSharpen.get() == 1:
        frame = img_sharp(frame)

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
    _, frame = vid.read()

    if effect.get() == "1":
        frame = img_mirror(frame)
    if effect.get() == "2":
        frame = img_kaleidoscope(frame)
    if effect.get() == "3":
        frame = img_swirl(frame)
    if effect.get() == "4":
        frame = img_light_tunnel(frame)

    if isSharpen.get() == 1:
        frame = img_sharp(frame)

    now = dt.now()
    filename = smplImgPath + 'image_' + now.strftime("%Y_%m_%d_%H_%M_%S") + '.jpg'
    cv2.imwrite(filename, frame)


# endregion

# region GUI
# Create a GUI app
app = Tk()

# Bind the app with Escape keyboard to
# quit app whenever pressed
app.bind('<Escape>', lambda e: app.quit())

# Create a label and display it on app
label_widget = Label(app)
label_widget.pack()

# Create button for image sharpening function
isSharpen = IntVar(master=app, value=0)
Checkbutton(app, text='Image sharpening', variable=isSharpen, onvalue=1, offvalue=0).pack()

# Create a button to save image
btnSave = Button(app, text="Capture", command=save_image)
btnSave.pack()

# Create radio button for effects
effect = StringVar(master=app, value="0")
for (text, value) in rValue.items():
    Radiobutton(app, text=text, variable=effect,
                value=value, indicator=0,
                background="light blue").pack(side=LEFT, expand=TRUE, fill=BOTH)
# endregion

# Open camera after the app displayed for 1 second
app.after(1000, open_camera())

# Create an infinite loop for displaying app on screen (center screen)
app.eval('tk::PlaceWindow . center')
app.mainloop()

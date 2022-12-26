# import Packages
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
import shutil
import tkinter.messagebox as msg
from tkinter import messagebox as mess
import cv2
import os
import csv
import numpy as np
import datetime
import time
import smtplib
import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from PIL import Image, ImageTk
from PIL import ImageTk, Image
import tensorflow as tf
from collections import deque
import numpy as np
import argparse
import cv2

path = ""
output_video = 'output.avi'
display = 1
size = 128
CLASSES = ["Cyclone", "Earthquake", "Flood", "Wildfire"]

print("[INFO] loading model and label binarizer...")
model = tf.keras.models.load_model('model.hdf5')

Q = deque(maxlen=size)

# Time function


def tick():
    time_string = time.strftime('%H:%M')
    clock.config(text=time_string)
    clock.after(200, tick)

# About Message Box


def about():
    mess._show(title='About us', message="This Project Is created by: \n (1) Member1 \n  "
                                         "Member2")


# Display Time and Date function
global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {'01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
        }

# Capture the Photo And save function
global res_label

def capture():
    print("[INFO] processing video...")
    filename = filedialog.askopenfilename(
        initialdir="/", title="Select A File", filetype=(("jpeg files", "*.mp4"), ("all files", "*.*"), ("jpeg files", "*.jpeg")))
    vs = cv2.VideoCapture(filename)
    writer = None
    (W, H) = (None, None)

    # loop over the video
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        # frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # clone the output frame, then convert it from BGR to RGB
        # ordering and resize the frame to a fixed 224x224
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype("float32")

        # prediction
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = CLASSES[i]
        file = open("result.txt","r+")
        file.truncate(0)
        file.close()
        with open('result.txt', 'w') as f:
            res = label
            f.writelines(res)

        # write on the output frame
        text = "activity: {}".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 255, 0), 5)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                output_video, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # write the output
        writer.write(output)

        if display > 0:
            # show the output image
            cv2.imshow("Output", output)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()


def send_email():
    name = simpledialog.askstring("Input", "What is your Location Name?")
    import smtplib
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("urva.surti@gmail.com", "Uhs18hds01vhs00$")
    f = open("result.txt", "r")
    #print(f.read())
    #message = "Calamity detected is : "+str(f.read()) + ". At location: " +str(name)+"."
    message=str(f.read())+" "+str(name)
    s.sendmail("urva.surti@gmail.com", "urva.surti@gmail.com", message)
    s.quit()

# GUI Interface
window = tk.Tk()
window.geometry("1235x650+50+30")
window.resizable(False, False)
window.title("Natural Calamities Detection")
window.configure(background='#262523')
p1 = PhotoImage(file='icon1.png')


# Setting icon of master window
#window.iconphoto(False, p1)

frame1 = tk.Frame(window, bg="#00aeff")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#00aeff")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="Natural Calamities Detection", fg="white", bg="#262523", width=55,
                    height=1, font=('times', 29, ' bold '))
message3.place(x=10, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

frame5 = tk.Frame(frame1, bg="white")
frame5.place(relx=0.06, rely=0.33, relwidth=0.90, relheight=0.60)

datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year + "  |  ", fg="orange", bg="#262523", width=80,
                 height=1, font=('times', 15, ' bold '))
datef.pack(fill='both', expand=1)

clock = tk.Label(frame3, fg="orange", bg="#262523", width=55,
                 height=1, font=('times', 15, ' bold '))
clock.pack(fill='both', expand=1)
tick()

head2 = tk.Label(frame2, text="                       Uploading Videos                             ", fg="black",
                 bg="#3ece48", font=('times', 17, ' bold '))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1, text="                       Natural Calamities                           ", fg="black",
                 bg="#3ece48", font=('times', 17, ' bold '))
head1.place(x=0, y=0)

message1 = tk.Label(frame2, text="*Dataset consists of four classes:Cyclone/hurricane", bg="#00aeff", fg="black", width=39,
                    height=1, activebackground="yellow", font=('times', 13, ' bold '))
message1.place(x=30, y=150)

message2 = tk.Label(frame2, text="*Earthquake, Flood, Wildfire", bg="#00aeff", fg="black", width=39,
                    height=1, activebackground="yellow", font=('times', 13, ' bold '))
message2.place(x=40, y=180)

# imgcap = tk.Label(frame2, text="Press Spacebar to Click Image", bg="#00aeff", fg="black", width=25,
#                     height=1, activebackground="yellow", font=('times', 13, ' bold '))
# imgcap.place(x=40, y=310)
# close = tk.Label(frame2, text="Press Backspace to Close", bg="#00aeff", fg="black", width=25,
#                     height=1, activebackground="yellow", font=('times', 13, ' bold '))
# close.place(x=20, y=340)


image = Image.open("out.jpg")
photo = ImageTk.PhotoImage(image.resize((450, 380), Image.ANTIALIAS))

label = Label(frame1, image=photo, bg='green')
label.image = photo
label.place(x=10, y=100)


menubar = tk.Menu(window, relief='ridge')
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='About Us', command=about)
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='About', font=('times', 29, ' bold '), menu=filemenu)

trainImg = tk.Button(frame2, text="Upload Videos", command=capture, fg="white", bg="blue", width=30, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=50, y=100)

capImg = tk.Button(frame2, text="Capture Images", command=capture, fg="white", bg="blue", width=30, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
capImg.place(x=50, y=250)


email = tk.Button(frame2, text="Send Email", command=send_email, fg="black", bg="yellow", width=30, height=1,
                  activebackground="Black", font=('times', 15, ' bold '))
email.place(x=50, y=400)


quitWindow = tk.Button(frame2, text="Quit", command=window.destroy, fg="black", bg="red", width=35, height=1,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=30, y=450)

window.configure(menu=menubar)
window.mainloop()

##################### END ######################################

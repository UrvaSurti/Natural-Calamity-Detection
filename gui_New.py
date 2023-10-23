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
from email.mime.multipart import MIMEMultipart
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
        initialdir="/", title="Select A File", filetype=(("jpeg files", "*.mp4"), ("all files", "*.*")))
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

        # # check if the video writer is None
        # if writer is None:
        #     # initialize our video writer
        #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #     writer = cv2.VideoWriter(
        #         output_video, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # # write the output
        # writer.write(output)


        cv2.imwrite('output.jpg',output)

        if display > 0:
            # show the output image
            cv2.imshow("Output", output)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # release the file pointers
    print("[INFO] cleaning up...")
    #writer.release()
    vs.release()
    f = open("result.txt", "r")
    result_calamity=str(f.read())


    if result_calamity =="Flood":
        message1.config(text="Precautions are as follows : Avoid driving through \n flooded areas and standing water.  ")
        message2.config(text=" If you evacuated, return to your home only after \n local authorities have said it is safe to do so.")
    elif result_calamity =="Cyclone":
        message1.config(text="Lock doors; turn off power, gas, and water; \n take your evacuation and emergency kits.")
        message2.config(text="If evacuating inland (out of town), take pets \n and leave early to avoid heavy flooding and wind hazards.")
    elif result_calamity =="Wildfire":
        message1.config(text="Precautions are as follows : Clear the area around \n the burn pile of any flammable debris.")
        message2.config(text=" Keep firefighting equipment handy a connected water \n hose or at least five gallons of water and a shovel should be nearby.")
    else:
        message1.config(text="Precautions are as follows : If you are in a car, pull \n over and stop. Set your parking brake. ")
        message2.config(text="If you are in bed, turn face down and cover your head \n and neck with a pillow.")
    mess._show(title='Clamity Detected', message="Predicted Disaster is : " + result_calamity)


def click():
    camera = cv2.VideoCapture(0)
    for i in range(1):
        return_value, image = camera.read()
        cv2.imwrite('img.jpg', image)
    del(camera)
    cv2.destroyAllWindows()
    print("[INFO] processing video...")
    vs = cv2.VideoCapture('img.jpg')
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
        # if writer is None:
        #     # initialize our video writer
        #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #     writer = cv2.VideoWriter(
        #         output_video, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # # write the output
        # writer.write(output)

        cv2.imwrite('output.jpg',output)

        if display > 0:
            # show the output image
            cv2.imshow("Output", output)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # release the file pointers
    print("[INFO] cleaning up...")
    #writer.release()
    vs.release()
    f = open("result.txt", "r")
    result_calamity=str(f.read())


    if result_calamity =="Flood":
        message1.config(text="Precautions are as follows : Avoid driving through \n flooded areas and standing water.  ")
        message2.config(text=" If you evacuated, return to your home only after \n local authorities have said it is safe to do so.")
    elif result_calamity =="Cyclone":
        message1.config(text="Lock doors; turn off power, gas, and water; \n take your evacuation and emergency kits.")
        message2.config(text="If evacuating inland (out of town), take pets \n and leave early to avoid heavy flooding and wind hazards.")
    elif result_calamity =="Wildfire":
        message1.config(text="Precautions are as follows : Clear the area around \n the burn pile of any flammable debris.")
        message2.config(text=" Keep firefighting equipment handy a connected water \n hose or at least five gallons of water and a shovel should be nearby.")
    else:
        message1.config(text="Precautions are as follows : If you are in a car, pull \n over and stop. Set your parking brake. ")
        message2.config(text="If you are in bed, turn face down and cover your head \n and neck with a pillow.")
    mess._show(title='Clamity Detected', message="Predicted Disaster is : " + result_calamity)


def send_email1():
    

    import smtplib
    from email.mime.text import MIMEText
    f = open("result.txt", "r")
    result_calamity=str(f.read())
    username = 'urva.surti@gmail.com'
    password = ''
    sender = 'urva.surti@gmail.com'
    targets = ['urva.surti@gmail.com']

    if result_calamity =="Flood":
        msg = MIMEText('Precautions are as follows : Avoid driving through flooded areas and standing water. As little as six inches of water can cause you to lose control of your vehicle. \n If you evacuated, return to your home only after local authorities have said it is safe to do so.')
    elif result_calamity =="Cyclone":
        msg = MIMEText('Lock doors; turn off power, gas, and water; take your evacuation and emergency kits. \n If evacuating inland (out of town), take pets and leave early to avoid heavy traffic, flooding and wind hazards.')
    elif result_calamity =="Wildfire":
        msg = MIMEText('Precautions are as follows : Clear the area around the burn pile of any flammable debris. \n Keep firefighting equipment handy - a connected water hose or at least five gallons of water and a shovel should be nearby.')
    else:
        msg = MIMEText('Precautions are as follows : If you are in a car, pull over and stop. Set your parking brake. \n If you are in bed, turn face down and cover your head and neck with a pillow.')

    msg['Subject'] = 'Precautions'
    msg['From'] = sender
    msg['To'] = "urva.surti@gmail.com"

    server = smtplib.SMTP("smtp.gmail.com:587")
    server.starttls()
    server.login(username, password)
    server.sendmail(sender, targets, msg.as_string())
    server.quit()


def send_email():
    name = simpledialog.askstring("Input", "What is your Location Name?")
    emailfrom = "urva.surti@gmail.com"
    emailto = "urva.surti@gmail.com"
    fileToSend = "output.jpg"
    username = "urva.surti@gmail.com"
    password = "Uhs18hds01vhs00$"

    f = open("result.txt", "r")
    
    msg = MIMEMultipart()
    msg["From"] = emailfrom
    msg["To"] = emailto
    msg["Subject"] = str(f.read()) +" Alert in "+ name 
    msg.preamble = str(f.read()) +"Alert"

    

    ctype, encoding = mimetypes.guess_type(fileToSend)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"

    maintype, subtype = ctype.split("/", 1)

    if maintype == "text":
        fp = open(fileToSend)
        # Note: we should handle calculating the charset
        attachment = MIMEText(fp.read(), _subtype=subtype)
        fp.close()
    else:
        fp = open(fileToSend, "rb")
        attachment = MIMEBase(maintype, subtype)
        attachment.set_payload(fp.read())
        fp.close()
        encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment", filename=fileToSend)
    msg.attach(attachment)

    server = smtplib.SMTP("smtp.gmail.com:587")
    server.starttls()
    server.login(username,password)
    server.sendmail(emailfrom, emailto, msg.as_string())
    server.quit()

    send_email1()
    print('Email Sent')

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
                    height=2, activebackground="yellow", font=('times', 13, ' bold '))
message1.place(x=30, y=170)

message2 = tk.Label(frame2, text="*Earthquake, Flood, Wildfire", bg="#00aeff", fg="black", width=39,
                    height=2, activebackground="yellow", font=('times', 13, ' bold '))
message2.place(x=40, y=230)

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

capImg = tk.Button(frame2, text="Upload Images", command=capture, fg="white", bg="blue", width=30, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
capImg.place(x=50, y=300)

capImg1 = tk.Button(frame2, text="Capture Images", command=click, fg="white", bg="blue", width=30, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
capImg1.place(x=50, y=350)


email = tk.Button(frame2, text="Send Email", command=send_email, fg="black", bg="yellow", width=30, height=1,
                  activebackground="Black", font=('times', 15, ' bold '))
email.place(x=50, y=400)


quitWindow = tk.Button(frame2, text="Quit", command=window.destroy, fg="black", bg="red", width=35, height=1,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=30, y=450)

window.configure(menu=menubar)
window.mainloop()

##################### END ######################################

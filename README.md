##  Natural calamities Detection

### Code Requirements
- Numpy(`pip install numpy==1.19.3`)
- sklearn(`pip install sklearn`)
- Matplotlib(`pip install matplotlib==3.0.1`)
- Tensorflow(`pip install tensorflow==1.12.0`)
- Keras(`pip install keras==2.2.4`)
- tk-tool(`pip install tk-tools`)
- Opencv(`pip install opencv-python`)
- Tkinter(Available in python)
- PIL (`pip install Pillow`)
- imutils(`pip install imutils`)
- h5py(`pip install h5py==2.8.0`)


### Other Requirements
- IP Webcam Application

### What steps you have to follow??
- Download my Repository 
- Open a `gui_New.py` and change the all paths with your system path
- Run `gui_New.py`.

****Note : You will have to change the directories according to your path, also have to change the database password and username.

### Project Structure

- We have trained a convolutional neural network
- The name of our trained model is Modelhdf.5
- We have also use a pre-trained model VGG16
- We have trained our model using 4000 images of different calamities including (flood , wildfire , earthquake, cyclone)
- Both these modules are combined
- We can upload or capture a image and upload on the computer based application
- It will analyse the image and predict the calamity
- It will also send necessary precautions via email to the natural calamity department of india

### Run
- Run the gui_New file then input all the data mentioned accordingly.
- Open command prompt
- Provide path to gui_New file 
- Type command "python gui_New.py"
- Upload video and images or capture images using the laptop webcam.


### Notes
- It will require high processing power(I have 8 GB RAM & 2 GB GC)
- Noisy image can reduce your accuracy so quality of images matter.



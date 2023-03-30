import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import cv2
import os
import random

#https://www.youtube.com/watch?v=BYvsdzHxXLA&ab_channel=TheWhiz - link to tutorial 

cred = credentials.Certificate(r"C:\Users\16134\OneDrive\Documents\Learning\Software\Projects\OpenCV Facial Detection/key.json") #path to key file 
app = firebase_admin.initialize_app(cred, {'storageBucket' : 'opencv-9e7cb.appspot.com'}) #showing where storge bucket is

bucket = storage.bucket()
blob = bucket.get_blob("IMG_E2587.JPG") #blob is the form that comes from firebase
arr = np.frombuffer(blob.download_as_string(), np.uint8) #converts from a blob, to a string to an array of bytes 
img = cv2.imdecode(arr, cv2.COLOR_BGR2BGR555) #actual image

cv2.imshow('image', img)

# path = r'C:\Users\16134\OneDrive\Documents\Learning\Software\Courses\OpenCV\Photos\Dharma/Dharma123.jpg'
# cv2.imwrite(path, img)

DIR = r'C:\Users\16134\OneDrive\Documents\Learning\Software\Courses\OpenCV\Photos'
name = (f'Dharma{random.randint(0,100)}')
person = (f'Dharma/{name}.jpg')
path = os.path.join(DIR, person)

print(path)
print(name)

cv2.waitKey(0)
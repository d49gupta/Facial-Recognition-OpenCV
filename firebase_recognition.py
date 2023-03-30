import numpy as np
import firebase_admin
from firebase_admin import credentials, storage
import cv2 as cv
import os 
import random

#Link to Firebase: https://console.firebase.google.com/

cred = credentials.Certificate(r"C:\Users\16134\OneDrive\Documents\Learning\Software\Projects\OpenCV Facial Detection/key.json") 
app = firebase_admin.initialize_app(cred, {'storageBucket' : 'opencv-9e7cb.appspot.com'}) 


people = []
for i in os.listdir(r'C:\Users\16134\OneDrive\Documents\Learning\Software\Courses\OpenCV\Photos'): 
    people.append(i)

haar_cascade = cv.CascadeClassifier(r'haar_face.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml') 

bucket = storage.bucket()
blob = bucket.get_blob("dharma1.jpg") 
arr = np.frombuffer(blob.download_as_string(), np.uint8) 
img = cv.imdecode(arr, cv.COLOR_BGR2BGR555) 

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
image = img.copy() #save a new image to save later

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4) 

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w] 

    label, confidence = face_recognizer.predict(faces_roi) 
    print(f'Label = {people[label]} with a confidence of {confidence}') 

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    
    confidence_level = (f'{confidence}%')

    cv.putText(img, confidence_level, (x + w ,y + h), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

#saves the image in the right folder to improve its training
DIR = r'C:\Users\16134\OneDrive\Documents\Learning\Software\Courses\OpenCV\Photos'
name = (f'{people[label]}{random.randint(0,100)}')
person = (f'{people[label]}/{name}.jpg')
path = os.path.join(DIR, person)

cv.imwrite(path, image)
print(path)
cv.waitKey(0)
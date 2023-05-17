import cv2
import numpy as np
import os,sqlite3
import face_recognition,pickle

path = "Encodings"
pathdata = "IMAGES"
dirlist = os.listdir(pathdata)
print(dirlist)
name="VAIBHAVI"
enc=[]
encodings=[]

image = cv2.imread(f'{pathdata}/{dirlist[3]}')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
encode = face_recognition.face_encodings(image)[0]

enc.append(encode)

f = open(f'{path}/{name}.pickle', "wb")

pickle.dump(encode,f)
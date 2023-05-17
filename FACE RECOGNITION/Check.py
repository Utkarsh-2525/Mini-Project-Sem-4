import numpy as np
import cv2,csv,pymongo
import pickle
import face_recognition
import os

enc=[]
with open('Encodings/utkarsh rai.csv', mode='r') as file:
    # reading the CSV file
    csvFile = csv.reader(file)

    for i in csvFile:

       enc.append(i)


data = {'name':'utkarsh','enc':enc[0]}

f = open("Encodings/encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
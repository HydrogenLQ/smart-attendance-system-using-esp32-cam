from urllib.request import urlopen
from datetime import datetime
import face_recognition
import pandas as pd
import numpy as np
import cvzone
import pickle
import csv
import cv2
import os


file_name = 'attendance.csv'
if os.path.exists(file_name):
    os.remove(file_name)
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Date'])

#encoding part starting ......

folderPath = r'B:\NewProject\image_folder'
pathlist = os.listdir(folderPath)
imgList = []
studentId = []
#spliting file name and extension
for path in pathlist:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentId.append(os.path.splitext(path)[0])
    
#img encoding finction
def fingEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)   
    return encodeList
encodeListKnown = fingEncodings(imgList)

#saving encodied img and file name 
encodeListID = [encodeListKnown, studentId]
file = open("EncodeImgFile.p",'wb')
pickle.dump(encodeListID, file)
file.close()

#encoding part ending ........


#attrndence part started ......      
def markAttendance(name):
    with open("Attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
        
#attendence part end....


#face recognition part start .....
#reading encoded img and file name
file = open("EncodeImgFile.p",'rb')
EncdImgId = pickle.load(file)
file.close()
EncImgList, Sid = EncdImgId 


url = r'http://192.168.0.150/capture'
cap = cv2.VideoCapture(url)
while True: 
    img_resp = urlopen(url)
    imgnp = np.asarray(bytearray(img_resp.read()), dtype="uint8")
    img = cv2.imdecode(imgnp, -1)
    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndx = np.argmin(faceDis)
        
        # print(matches)
        # print(faceDis)
        # print(matchIndx)
        
        if matches[matchIndx]:
            print("Welcom Mr : ", end=" ")
            print(Sid[matchIndx])
            markAttendance(Sid[matchIndx])


    cv2.imshow("Camera", img)
    if cv2.waitKey(1) == 113:
        break
#remove duplicates row
Attendance = pd.read_csv("attendance.csv")
Attendance.drop_duplicates(subset = "ID", keep = 'first', inplace = True)
uniqueId = Attendance
file_name = 'attendance.csv'
os.remove(file_name)
uniqueId.to_csv('attendance.csv')
import cv2
import numpy
import os

bus_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'Bus_front.xml')
datasets = 'datasets'
print('training...')
(image, labels, name, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        name[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            image.append(cv2.imread(path, 0))
            labels.append(int(label))
        id +=1

(images, labels) =[numpy.array(lis) for lis in [image, labels]]
print(image, labels)
(width, height) = (130, 100)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(image, labels)

webcam = cv2.VideoCapture(0)
cnt=0

(_, im) = webcam.read()

def detect_bus(gray):


    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = bus_detect.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0, ), 3)
        if prediction[1]<800:
            cv2.putText(im, '%s - %.0f' % (name[prediction[0]], prediction[1]), (x - 10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),2)
            print (name[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(im,'unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 0))

while True:
    if(cnt>100):
                print("unknown Person")
                cv2.imread("unknown.jpg",im)
                cnt=0
    cv2.imshow('FaceRecognition', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()



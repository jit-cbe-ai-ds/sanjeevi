import cv2, os

datasets = 'datasets'
sub_data = 'sanj'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)

bus_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'Bus_front.xml')
webcam = cv2.VideoCapture(0)

(_, im) = webcam.read()


def detect_bus(gray):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bus1 = bus_detect.detectMultiScale(gray, 1.50, 4)
    for (x, y, w, h) in bus1:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        bus = gray[y:y + h, x:x + w]
        bus_resize = cv2.resize(bus, (width, height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize)


count = 1
while count < 30:
    count += 1
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

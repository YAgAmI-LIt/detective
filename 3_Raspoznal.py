import cv2
import numpy as np

# создаем объект распознавателя и загружаем данные обучения
# recognizer = cv2.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.load('trainner/trainner.yml')
recognizer.read('trainner/trainner.yml')
# создадим каскадный классификатор, используя каскад хаара для обнаружения лиц,
# предполагая, что у вас есть файл каскада в том же месте
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
# нужен шрифт для текста
font = cv2.FONT_HERSHEY_SIMPLEX
Id = 0
names = ['None', 'Elena', 'Igor', 'Lenochka', 'Lida', 'Natacha', 'Andrey', 'Gleb', 'Larisa', 'Z', 'W']
# создадим объект захвата видео
cam = cv2.VideoCapture(0)
cam.set(3,640) # ширина кадра
cam.set(4,480) # set высота
# Определит минимальный размер окна для распознавания лица
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(minW), int(minH)))
    # Для каждого лица в лицах
    for(x,y,w,h) in faces:
        # Создайте прямоугольник вокруг лица
        # cv2.rectangle(im,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),4)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Распознать лицо, которому принадлежит ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        #  Проверит вероятность, что они 100 ==> "0" идеально подходит
        #if (confidence & lt;50):
        if (confidence) < 100:
            Id = names[Id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(im, str(Id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(im, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    # Отобразить видеокадр с ограниченным прямоугольником
    cv2.imshow('im',im)
    # Если нажать 'q', закрыть программу
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
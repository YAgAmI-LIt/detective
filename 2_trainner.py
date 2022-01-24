import cv2  #добавила библиотеку opencv-contrib-python

import os
import numpy as np
from PIL import Image

# инициализировать распознаватель и детектор лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()   # добавила библиотеку opencv-contrib-python
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # Для загрузки приложения получим пути ко всем папкам и файлам
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # Создадим 2 индификатор (пустых списка)
    faceSamples = []
    Ids = []
# перебираем все пути к изображениям и загружаем идентификаторы и изображения
    for imagePath in imagePaths:
        # игнорировать, если файл не имеет расширения jpg
        if (os.path.split(imagePath)[-1].split(".")[-1] != 'jpg'):
            continue
        # загрузка изображения и преобразование его в шкалу серого
        pilImage = Image.open(imagePath).convert('L')
        # конвертируем изображение PIL в массив numpy
        imageNp = np.array(pilImage, 'uint8')
        # получение идентификатора из изображения
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # извлечь лицо из образца обучающего изображения
        faces = detector.detectMultiScale(imageNp)
        # Если есть лицо, добавьте его в список вместе с его идентификатором.
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids
# вызвать эту функцию и передать данные в распознаватель для обучения.
faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')

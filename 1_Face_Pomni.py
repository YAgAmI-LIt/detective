import cv2
import os

vid_cam = cv2.VideoCapture(0)
vid_cam.set(3,640) # ширина кадра
vid_cam.set(4,480) # set высота
# Обнаружение объекта в видеопотоке с помощью Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Для каждого человека один идентификатор лица
face_id = input('\n Bведите идентификатор пользователя и нажмите  ==>  ')
print("\n [INFO] Инициализация захвата лица.  Посмотри в камеру и подожди ...")
# Инициализировать образец изображения лица
count = 0
while True:
    # Захватить кадр видео
    ret, image_frame = vid_cam.read()
    # Преобразовать кадр в оттенки серого
    grey = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    # Обнаружение рамок разного размера, список прямоугольников лиц
    faces = face_detector.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    # Цикл для каждого лица
    for (x,y,w,h) in faces:
        # Обрезать рамку изображения до прямоугольника
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), 255,0,0, 2)
        roi_gray = grey[y:y + h, x:x + w]
        roi_color = image_frame[y:y + h, x:x + w]
        # увеличивающийся номер образца
        count += 1
        # сохранение захватываемого лица в папке набор
        cv2.imwrite("dataSet/User." + str(face_id) + "." + str(count) + ".jpg", grey[y:y+h,x:x+w])
        # Отображение кадра видео с ограниченным прямоугольником на лице человека
        cv2.imshow('frame', image_frame)
        # Чтобы прекратить съемку видео, нажмите q и удерживайте не менее 100 мс.
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # Если количество снимков достигло 30, прекратите снимать видео.
    elif count > 99:
        break
print ("\ n [INFO] Выход из программы и очистка")
# Остановить видео
vid_cam.relea8se()
# Закройте все запущенные окна
cv2.destroyAllWindows()

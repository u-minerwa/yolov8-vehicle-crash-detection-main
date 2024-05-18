import cv2

# Открываем видеофайл
cap = cv2.VideoCapture('pm.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    # Проверяем, получили ли мы кадр
    if ret:
        cv2.imshow('Video', frame)

        # Ждем 25 миллисекунд и ожидаем нажатия клавиши
        key = cv2.waitKey(25)

        # Если нажата клавиша 'p' (в данном случае код 112), ставим видео на паузу
        if key == 112:  
            while True:
                # Ждем нажатия клавиши
                key = cv2.waitKey(25)
                # Если нажата клавиша 'p' - возобновляем воспроизведение
                if key == 112:
                    break

    else:
        break

cap.release()
cv2.destroyAllWindows()


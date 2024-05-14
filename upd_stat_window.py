import numpy as np
import cv2

def update_statistics_window(statistics):
    # Создаём изображение для отображения статистики
    stat_image = 255 * np.ones((200, 300, 3), dtype=np.uint8)  # Белое изображение размером 200x300 

    # Добавляем текст статистики на изображение
    cv2.putText(stat_image, f"Accident: {statistics['Accident']}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
    cv2.putText(stat_image, f"TrafficLight: {statistics['TrafficLight']}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
    cv2.putText(stat_image, f"Car: {statistics['Car']}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
    cv2.putText(stat_image, f"Sign: {statistics['Sign']}", (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2) 

    # Отображаем изображение статистики
    cv2.imshow("Statistics", stat_image)
    cv2.waitKey(1)


# Создаём окно для статистики
# cv2.namedWindow("Statistics") 
#update_statistics_window(statistics)


# Сохраняем кадр с ДТП в файл с датой и временем в названии
# cv2.imwrite(f"AccidentFrames/accident_frame_{dt_string}_{dtp_count}.png", frame)

# Сохраняем кадр с ДТП в файл
# cv2.imwrite(f"AccidentFrames/accident_frame_{dtp_count}.png", frame)
# cv2.imwrite("AccidentFrames/accident_frame.png", frame)


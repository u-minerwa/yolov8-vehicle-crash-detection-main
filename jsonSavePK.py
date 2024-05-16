import os
import json
from datetime import datetime

# Путь к папке для сохранения файлов
save_folder = "path/to/your/folder"

# Переменная, которая будет хранить порядковый номер
file_counter = 1

# Генерируем имя файла на основе текущей даты, времени и порядкового номера
def generate_file_name():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    return os.path.join(save_folder, f"data_{dt_string}_{file_counter}.json")

# Создаем словарь с нужными данными
data_to_save = {
    "Statistics": {
        "Accident": statistics['Accident'],
        "TrafficLight": statistics['TrafficLight'],
        "Car": statistics['Car'],
        "Sign": statistics['Sign'],
        "TotalAccidents": total_accident_frames
    },
    "DateTime": dt_string
}

# Получаем имя файла
file_name = generate_file_name()

# Сохраняем данные в файл JSON
with open(file_name, 'w') as json_file:
    json.dump(data_to_save, json_file)

print("Данные успешно сохранены в файл", file_name)

# Увеличиваем порядковый номер для следующего файла
file_counter += 1

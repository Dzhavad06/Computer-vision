
import cv2  # Для работы с видеопотоком и изображениями
import face_recognition  # Для распознавания лиц и получения векторов
import sqlite3  # Управление SQLite базой данных
import json  # Сериализация/десериализация векторов лиц
import numpy as np  # Для работы с массивами и вычисления расстояний
from datetime import datetime  # Для отметок времени
import time  # Для интервалов времени

TOLERANCE = 0.6          # Максимальное расстояние для распознавания (чем меньше, тем строже)
LOG_INTERVAL = 5         # Минимальное время (сек) между записями одного человека
SCALE = 0.25             # Масштаб кадра для ускорения обработки

conn = sqlite3.connect("../faces.db")  # Подключение к базе
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS recognition_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date_time TEXT
)
""")  # Создаём таблицу журнала распознаваний, если её нет
conn.commit()

cursor.execute("""
SELECT users.name, face_vectors.vector
FROM face_vectors
JOIN users ON face_vectors.user_id = users.id
""")  # Получаем имена и векторы лиц из базы

known_names = []  # Список имён известных пользователей
known_vectors = []  # Список векторов лиц

for name, vector_json in cursor.fetchall():  # Загружаем данные из запроса
    known_names.append(name)
    known_vectors.append(np.array(json.loads(vector_json)))

print(f"Загружено векторов: {len(known_vectors)}")  # Кол-во загруженных векторов

cap = cv2.VideoCapture(0)  # Открываем камеру
if not cap.isOpened():
    print("Не удалось открыть камеру.")
    conn.close()
    raise SystemExit(1)

print("Камера запущена. Нажмите Q для выхода.")

last_logged = {}  # Хранит время последней записи для каждого имени

while True:
    ret, frame = cap.read()  # Считываем кадр
    if not ret:
        print("Ошибка получения кадра.")
        break

    # Уменьшаем кадр, чтобы ускорить вычисления
    small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)  # Переводим в RGB для face_recognition

    face_locations = face_recognition.face_locations(rgb_small)  # Поиск лиц
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)  # Векторы лиц

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"  # Имя по умолчанию

        if known_vectors:
            distances = face_recognition.face_distance(known_vectors, face_encoding)  # Расстояния
            best = int(np.argmin(distances))  # Индекс ближайшего по вектору лица
            if distances[best] <= TOLERANCE:
                name = known_names[best]  # Подбираем имя

        # Логирование нового распознавания с интервалом
        if name != "Unknown":
            now = time.time()
            if now - last_logged.get(name, 0) >= LOG_INTERVAL:
                date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "INSERT INTO recognition_log (name, date_time) VALUES (?, ?)",
                    (name, date_time),
                )
                conn.commit()
                last_logged[name] = now  # Обновляем время последней записи
                print(f"Распознан: {name} | Время: {date_time}")

        # Возвращаем координаты рамки к исходному размеру кадра
        k = int(1 / SCALE)
        top, right, bottom, left = top * k, right * k, bottom * k, left * k

        # Рисуем рамку и имя на исходном кадре
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face recognition", frame)  # Показываем кадр с рамками

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Выход по нажатию Q
        break

cap.release()  # Освобождаем камеру
cv2.destroyAllWindows() # Закрываем окна conn.close() # Закрываем соединение с базой
conn.close()



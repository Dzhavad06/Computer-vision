import cv2  # Импорт OpenCV для работы с камерой и изображениями
import face_recognition  # Библиотека для распознавания лиц
import sqlite3  # Работа с SQLite базой
import json  # Для преобразования векторов в строки

NAME = input("Введите имя человека: ")  # Ввод имени пользователя

conn = sqlite3.connect("faces.db")  # Подключаемся к базе данных
cursor = conn.cursor()  # Получаем курсор для выполнения запросов

cursor.execute("""CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
)""")  # Создаём таблицу пользователей если нет

cursor.execute("""CREATE TABLE IF NOT EXISTS face_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    vector TEXT
)""")  # Создаём таблицу для хранения векторов лиц

conn.commit()  # Сохраняем изменения в базе

cursor.execute("INSERT INTO users (name) VALUES (?)", (NAME,))  # Вставляем имя в таблицу users
user_id = cursor.lastrowid  # Получаем id нового пользователя
conn.commit()  # Сохраняем изменения

cap = cv2.VideoCapture(0)  # Запускаем камеру

print("Нажмите S, чтобы начать сбор 10 векторов.")  # Инструкция пользователю
print("Для выхода нажмите Q.")  # Инструкция для выхода

vectors_count = 0  # Счётчик сохранённых векторов
collecting = False  # Флаг начала сбора векторов

while True:  # Основной цикл
    ret, frame = cap.read()  # Считываем кадр с камеры
    if not ret:  # Если кадр не получен
        print("Ошибка получения кадра с камеры.")  # Вывод ошибки
        break  # Выходим из цикла

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Конвертируем BGR в RGB
    face_locations = face_recognition.face_locations(rgb_frame)  # Находим лица на кадре

    for top, right, bottom, left in face_locations:  # Для каждого лица
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Рисуем рамку

    cv2.putText(frame, f"Vectors: {vectors_count}/10", (20, 40),  # Показываем счётчик
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face registration", frame)  # Показываем окно с видео

    key = cv2.waitKey(1) & 0xFF  # Считываем нажатие клавиш

    if (key == ord("s") or key == ord("ы")) and not collecting:  # Если нажата S и сбор не начат
        collecting = True  # Запускаем сбор векторов
        print("Начался сбор векторов...")  # Сообщение

    if collecting and vectors_count < 10:  # Если сбор активен и нужно ещё векторы
        if len(face_locations) == 1:  # Если ровно одно лицо
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)  # Берём вектор
            if encodings:  # Если вектор есть
                vector = encodings[0]  # Первый вектор лица
                cursor.execute(
                    "INSERT INTO face_vectors (user_id, vector) VALUES (?, ?)",  # Вставляем в базу
                    (user_id, json.dumps(vector.tolist()))
                )
                conn.commit()  # Сохраняем изменения
                vectors_count += 1  # Увеличиваем счётчик
                print(f"Сохранено: {vectors_count}/10")  # Вывод прогресса
                cv2.waitKey(1000)  # Пауза 1 секунда для сбора следующего кадра
        else:  # Если лиц нет или больше одного
            print("Нужно, чтобы в кадре было ровно одно лицо.")  # Вывод предупреждения

    if vectors_count >= 10:  # Если собрано 10 векторов
        collecting = False  # Останавливаем сбор
        print("Сбор завершён! Получено 10 векторов.")  # Сообщение
        print("Нажмите Q для выхода.")  # Подсказка выхода

    if key == ord("q") or key == ord("й"):  # Если нажата Q
        break  # Выходим из цикла

cap.release()  # Освобождаем камеру
cv2.destroyAllWindows()  # Закрываем все окна OpenCV
conn.close()  # Закрываем соединение с базой




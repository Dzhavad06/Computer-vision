import cv2
import face_recognition
import sqlite3
import json

# 1. Запрашиваем имя и создаём базу данных
NAME = input("Введите имя человека: ")

conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS face_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    vector TEXT
)
""")

conn.commit()

# Добавляем пользователя
cursor.execute("INSERT INTO users (name) VALUES (?)", (NAME,))
user_id = cursor.lastrowid
conn.commit()

# 2. Запускаем камеру
cap = cv2.VideoCapture(0)

print("Нажмите S, чтобы начать сбор 10 векторов.")
print("Для выхода нажмите Q.")

vectors_count = 0
collecting = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка получения кадра с камеры.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    # Рисуем рамки вокруг найденных лиц
    for top, right, bottom, left in face_locations:
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Vectors: {vectors_count}/10",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face registration", frame)

    key = cv2.waitKey(1) & 0xFF

    # Начинаем сбор
    if (key == ord("s") or key == ord("ы")) and not collecting:
        collecting = True
        print("Начался сбор векторов...")

    # Сбор 10 векторов
    if collecting and vectors_count < 10:

        # Сохраняем только если в кадре ровно одно лицо
        if len(face_locations) == 1:

            encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations
            )

            if encodings:
                vector = encodings[0]

                cursor.execute(
                    """
                    INSERT INTO face_vectors (user_id, vector)
                    VALUES (?, ?)
                    """,
                    (
                        user_id,
                        json.dumps(vector.tolist())
                    )
                )

                conn.commit()

                vectors_count += 1
                print(f"Сохранено: {vectors_count}/10")

                # Небольшая пауза между векторами
                cv2.waitKey(1000)

        else:
            print("Нужно, чтобы в кадре было ровно одно лицо.")

    # Закончили сбор
    if vectors_count >= 10:
        collecting = False
        print("Сбор завершён! Получено 10 векторов.")
        print("Нажмите Q для выхода.")

    # Выход
    if key == ord("q") or key == ord("й"):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()




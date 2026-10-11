import cv2                          # импортируем OpenCV для работы с видео и изображениями
import face_recognition             # импортируем библиотеку для распознавания лиц

cap = cv2.VideoCapture(0)          # открываем камеру с индексом 0

if not cap.isOpened():             # проверяем, открылась ли камера
    print("Камера не открылась")   # если нет - выводим сообщение
    exit()                        # завершаем программу

print("Камера запущена. Нажмите Q для выхода.")  # информируем пользователя

while True:
    ret, frame = cap.read()       # читаем кадр с камеры

    if not ret:                  # если кадр не получен
        print("Не удалось получить кадр")  # сообщаем об ошибке
        break                   # выходим из цикла

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # конвертируем BGR (OpenCV) в RGB (face_recognition)

    face_locations = face_recognition.face_locations(rgb_frame)  # находим координаты лиц на кадре

    for top, right, bottom, left in face_locations:    # для каждого лица
        cv2.rectangle(                                  # рисуем зелёную прямоугольную рамку
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

    cv2.putText(                              # выводим текст с количеством лиц на кадре
        frame,
        f"Faces: {len(face_locations)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face detection", frame)      # показываем обработанный кадр

    if cv2.waitKey(1) & 0xFF == ord("q"):   # если нажата клавиша Q — выходим из цикла
        break

cap.release()                              # освобождаем камеру
cv2.destroyAllWindows()                    # закрываем все окна OpenCV
import cv2
import face_recognition

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Камера не открылась")
    exit()

print("Наведите лицо на камеру.")
print("Нажмите S, чтобы получить вектор.")
print("Нажмите Q для выхода.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face encoding test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if len(face_locations) == 0:
            print("Лицо не найдено.")
            continue

        encodings = face_recognition.face_encodings(
            rgb_frame,
            face_locations
        )

        vector = encodings[0]

        print("Вектор получен!")
        print("Размер вектора:", len(vector))
        print("Первые 10 значений:", vector[:10])

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
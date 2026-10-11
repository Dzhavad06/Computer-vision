import cv2
import face_recognition

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Камера не открылась")
    exit()

print("Камера запущена. Нажмите Q для выхода.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр")
        break

    # OpenCV использует BGR, face_recognition ожидает RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ищем лица
    face_locations = face_recognition.face_locations(rgb_frame)

    # Рисуем рамку вокруг каждого найденного лица
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
        f"Faces: {len(face_locations)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
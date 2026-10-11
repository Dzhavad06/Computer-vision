"""
КТ по трекингу объектов: YOLO (готовая модель) + ByteTrack + Supervision.
Считает, сколько объектов пересекло линию, и сохраняет видео tracked.mp4.

Запуск:
    python tracking.py video.mp4
    python tracking.py 0            # веб-камера
"""
import sys
import cv2
import supervision as sv
from ultralytics import YOLO

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
if SOURCE.isdigit():
    SOURCE = int(SOURCE)

OUTPUT = "tracked.mp4"
CLASSES = [0]          # классы COCO: 0 = человек, 2 = машина, 5 = автобус, 7 = грузовик
CONF = 0.25
IMGSZ = 1280           # больше = лучше находит мелких людей вдали но медленнее
MAX_WIDTH = 1280       # 4K-кадры уменьшаем до этой ширины

# Линия подсчёта
LINE_VERTICAL = False  # True - вертикальная, False - горизонтальная
LINE_POS = 0.70        # положение линии: доля от высоты (или ширины) кадра

model = YOLO("yolov8n.pt")   # готовая обученная модель (скачается сама)

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Не удалось открыть источник: {SOURCE}")

src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

scale = min(1.0, MAX_WIDTH / src_w)
w, h = int(src_w * scale), int(src_h * scale)

if LINE_VERTICAL:
    x = int(w * LINE_POS)
    line_zone = sv.LineZone(
        start=sv.Point(x, 0), end=sv.Point(x, h),
        triggering_anchors=[sv.Position.CENTER],
    )
else:
    y = int(h * LINE_POS)
    line_zone = sv.LineZone(
        start=sv.Point(0, y), end=sv.Point(w, y),
        triggering_anchors=[sv.Position.CENTER],
    )

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
line_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=1)

writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if scale != 1.0:
        frame = cv2.resize(frame, (w, h))

    # model.track с ByteTrack присваивает каждому объекту постоянный id
    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=CLASSES,
        conf=CONF,
        imgsz=IMGSZ,
        verbose=False,
    )[0]

    detections = sv.Detections.from_ultralytics(result)

    if detections.tracker_id is not None and len(detections) > 0:
        line_zone.trigger(detections)

        labels = [
            f"#{tid} {model.names[int(cid)]}"
            for cid, tid in zip(detections.class_id, detections.tracker_id)
        ]
        frame = trace_annotator.annotate(frame, detections)
        frame = box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections, labels)

    frame = line_annotator.annotate(frame, line_counter=line_zone)
    cv2.putText(
        frame,
        f"In: {line_zone.in_count}  Out: {line_zone.out_count}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
    )

    writer.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Готово. In: {line_zone.in_count}, Out: {line_zone.out_count}. Видео: {OUTPUT}")


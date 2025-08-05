import cv2
import torch
from gtts import gTTS
import os
import time

# โหลดโมเดล YOLOv5n (Nano – เบาและเร็ว)
model = torch.hub.load('yolov5', 'yolov5n', source='local')

# เปิดกล้อง 720p
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_label = ""
last_time = 0
frame_count = 0
skip_frame = 2  # ตรวจจับทุก 2 เฟรม

# คำแปลชื่อวัตถุ
label_dict = {
    "apple": "แอปเปิล",
    "bottle": "ขวดน้ำ",
    "sports ball": "ลูกบอล",
    "banana": "กล้วย"
    # ไม่มี "person" ใน dict = ไม่พูด
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    if frame_count % skip_frame == 0:
        results = model(frame)
        detections = results.pred[0]
        labels = results.names

        for *box, conf, cls in detections:
            label = labels[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # พูดเฉพาะ label ที่ไม่ใช่ "person"
            if label in label_dict:
                if label != last_label or time.time() - last_time > 3:
                    speak_label = label_dict[label]
                    print(f'พูดว่า: {speak_label}')
                    tts = gTTS(text=speak_label, lang='th')
                    tts.save('speak.mp3')
                    os.system('start speak.mp3')
                    last_label = label
                    last_time = time.time()

    cv2.imshow('Object Detection 720p', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

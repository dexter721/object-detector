import cv2
import torch
from gtts import gTTS
import os
import time
from playsound import playsound
import threading

# โหลดโมเดล YOLOv5n (Nano – เบาและเร็ว)
model = torch.hub.load('yolov5', 'yolov5n', source='local')

# เปิดกล้อง 720p
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
skip_frame = 2  # ตรวจจับทุก 2 เฟรม

# คำแปลชื่อวัตถุ
label_dict = {
    "apple": "แอปเปิล",
    "bottle": "ขวดน้ำ",
    "sports ball": "ลูกบอล",
    "banana": "กล้วย"
}

# เก็บเวลาพูดครั้งล่าสุดของแต่ละวัตถุ
last_spoken = {}

# ฟังก์ชันเล่นเสียงแบบไม่บล็อก
def play_sound(file):
    threading.Thread(target=playsound, args=(file,), daemon=True).start()

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

            # พูดเฉพาะวัตถุที่อยู่ใน label_dict เท่านั้น
            if label in label_dict:
                current_time = time.time()
                last_time = last_spoken.get(label, 0)

                # พูดก็ต่อเมื่อยังไม่เคยพูด หรือผ่านมาแล้วเกิน 10 วินาที
                if current_time - last_time > 10:
                    speak_label = label_dict[label]
                    print(f'พูดว่า: {speak_label}')
                    tts = gTTS(text=speak_label, lang='th')
                    tts.save('speak.mp3')
                    play_sound('speak.mp3')
                    last_spoken[label] = current_time

    cv2.imshow('Object Detection 720p', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import torch
from gtts import gTTS
import os
import time
from playsound import playsound
import threading

model = torch.hub.load('yolov5', 'yolov5n', source='local')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
skip_frame = 5  # ตรวจจับทุก 5 เฟรม
speak_cooldown = 5  # พูดห่างกัน 5 วินาที

last_label = ""
last_time_spoken = 0
is_speaking = False

label_dict = {
    "apple": "แอปเปิล",
    "bottle": "ขวดน้ำ",
    "sports ball": "ลูกบอล",
    "banana": "กล้วย"
}

def speak(text):
    global is_speaking
    is_speaking = True
    tts = gTTS(text=text, lang='th')
    tts.save('speak.mp3')
    playsound('speak.mp3')
    is_speaking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    if frame_count % skip_frame == 0 and not is_speaking:
        results = model(frame)
        detections = results.pred[0]
        labels = results.names

        for *box, conf, cls in detections:
            label = labels[int(cls)]

            if label in label_dict:
                now = time.time()
                # พูดเฉพาะถ้า: ไม่ใช่วัตถุเดิม หรือผ่านไปแล้วนานพอ
                if label != last_label or now - last_time_spoken > speak_cooldown:
                    last_label = label
                    last_time_spoken = now
                    text_to_speak = label_dict[label]
                    print(f'พูดว่า: {text_to_speak}')
                    threading.Thread(target=speak, args=(text_to_speak,)).start()
                break  # เจอแล้วพูดตัวเดียวพอ

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

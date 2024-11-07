import cv2
import threading
from queue import Queue
from ultralytics import YOLO

model = YOLO("rugby.pt")


cap = cv2.VideoCapture('match trimmed.mp4')


target_width = 640
target_height = 480

frame_queue = Queue(maxsize=1000000)
result_queue = Queue(maxsize=1000000)


def process_frames():
    while True:
        if not frame_queue.empty():
            frame, frame_width = frame_queue.get()
            frame_resized = cv2.resize(frame, (target_width, target_height))
            results = model(frame_resized)
            
            left_border = frame_width // 2 - frame_width // 5
            right_border = frame_width // 2 + frame_width // 5
            
 
            # cv2.line(frame_resized, (left_border, 0), (left_border, frame_resized.shape[0]), (0, 255, 0), 2)
            # cv2.line(frame_resized, (right_border, 0), (right_border, frame_resized.shape[0]), (0, 255, 0), 2)
            
            for result in results:
                for box in result.boxes:
                    x, y, width, height = box.xywh[0].tolist()
                    x_center = x
                    
                    if x_center < left_border:
                        print("left")
                    elif x_center > right_border:
                        print("right")
                    else:
                        print("center")
            annotated_frame = results[0].plot()
            result_queue.put(annotated_frame)

threading.Thread(target=process_frames, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_width = frame.shape[1]
    frame_queue.put((frame, frame_width))
    
    if not result_queue.empty():
        processed_frame = result_queue.get()
        cv2.imshow('YOLO Live Inference', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

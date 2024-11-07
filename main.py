import cv2
from ultralytics import YOLO

model = YOLO("rugby.pt") 
cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()  
    if not ret:
        break
    
    results = model(frame)
    frame_width = frame.shape[1]
    
    # Define the borders for the center region
    left_border = frame_width // 2 - frame_width // 5
    right_border = frame_width // 2 + frame_width // 56


    # cv2.line(frame, (left_border, 0), (left_border, frame.shape[0]), (0, 255, 0), 2) 
    # cv2.line(frame, (right_border, 0), (right_border, frame.shape[0]), (0, 255, 0), 2)  
    
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
    cv2.imshow('YOLO Live Inference', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

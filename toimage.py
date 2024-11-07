import cv2
import os

def capture_screenshots(video_path, output_folder, interval=25):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        if frame_count % interval == 0:
            screenshot_filename = os.path.join(output_folder, f"screenshot_{screenshot_count}.png")
            cv2.imwrite(screenshot_filename, frame)
            screenshot_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Captured {screenshot_count} screenshots and saved to {output_folder}")

video_path = 'match trimmed.mp4'
output_folder = 'output_screenshots'
capture_screenshots(video_path, output_folder)

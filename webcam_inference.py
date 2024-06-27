import cv2
from ultralytics import YOLO

model = YOLO('D:/runs/pose/train2/weights/last.pt')

video_path = 0
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Resize the frame to 384x640
        resized_frame = cv2.resize(frame, (640, 384))
        
        results = model(resized_frame, save=True)
        annotated_frame = results[0].plot()
        
        cv2.imshow('Annotation', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

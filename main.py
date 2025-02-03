import cv2
from ultralytics import YOLO

#Import Trained YOLO Model
model = YOLO("best.pt")

#Open webcam
for result in model(source=0, show=True, conf=0.6, stream=True):  
    boxes = result.boxes  # Get  boxes

    if boxes is not None:  
        for box in boxes:
            confidence = box.conf.item()  # Get Confidence Score
            
            print(f"Confidence: {confidence:.2f}")  # Print confidence score
            
            # If confidence is greater than 0.9, display message
            if confidence > 0.9:
                print("GET OFF YOUR PHONE")
    if cv2.waitKey(1) & 0xFF == ord('e'):  
        break   

cv2.destroyAllWindows()  
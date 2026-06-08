import cv2 as cv 
import numpy as np  
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
# results = model.train(data="mnist160", epochs=1)
# metrics = model.val()

# metrics.box.map
# metrics.box.map50
# metrics.box.map75
# metrics.box.maps
# metrics.box.image_metrics


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read Frame")
        break
    
    results = model.predict(frame)

    annotated_frame = frame
    for result in results:
        # xywh = result.boxes.xywh
        # xywhn = result.boxes.xywhn
        # xyxy = result.boxes.xyxy
        # xyxyn = result.boxes.xyxyn
        # names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        # confs = result.boxes.conf
        # annotated_frame = result.plot()

        if result.probs is not None:
            # 1. Get the index of the highest probability class
            top1_idx = result.probs.top1
            
            # 2. Get the confidence score (convert PyTorch tensor to a float)
            confidence = result.probs.top1conf.item()
            
            # 3. Look up the string name using the class index
            class_name = result.names[top1_idx]
            
            # 4. Format the string (e.g., "Class: 94%")
            display_text = f"{class_name}: {confidence * 100:.1f}%"
            
            # 5. Draw the text onto your live frame
            cv.putText(
                img=frame,
                text=display_text,
                org=(30, 50),               # Coordinates (X, Y) to place text
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,              # Font size
                color=(0,0 ,255),          # BGR color (Green)
                thickness=2,
                lineType=cv.LINE_AA
            )

        

    cv.imshow("Frame", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

cv.destroyAllWindows()
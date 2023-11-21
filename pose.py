import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")  # load a pretrained model

# Define the video capture device
cap = cv2.VideoCapture('LNNX2982.mp4')


while True:
    print("Frame read successfully")
    # Read a frame from the video stream
    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    
    # Stop the program if no frame is read
    if not ret:
        break
    
    # Use the model to predict the poses in the frame
    results = model.predict(frame,save=True)
    print('results',results)
    detection = results[0].plot()
    # Display the resulting image with the predicted poses
    cv2.imshow('YOLOv8 Pose Detection', detection)
    
    # Exit the program if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close the display window
cap.release()
cv2.destroyAllWindows()
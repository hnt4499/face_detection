# Import OpenCV
import cv2
# Loading the cascades
face_cascade =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# This function is to detect faces and eyes, in which eyes detection will be done inside the
# face frame to minimize processing time.
def detect(gray_frame, original_frame):
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (255 , 0, 0), 2)
        face_rectangle_gray = gray_frame[y : y + h, x : x + w]
        face_rectangle_color = original_frame[y : y + h, x : x + w]
        # Detecting eyes inside the zone of face detected.
        eyes = eye_cascade.detectMultiScale(face_rectangle_gray, 1.1, 3)
        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            cv2.rectangle(face_rectangle_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 1)
    return original_frame

# Initializing the webcam
video_capture = cv2.VideoCapture(0)
while True:
    original_frame = video_capture.read()[1]
    gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray_frame, original_frame)
    cv2.imshow("Face Detection", canvas)
    # Press e to exit the detection. Otherwise, it will be a infinite loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


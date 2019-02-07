import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

def detect(original_frame):
    gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        face_frame_gray = gray_frame[y : y + h, x : x + w]
        face_frame_color = original_frame[y : y + h, x : x + w]
        smiles = smile_cascade.detectMultiScale(face_frame_gray, 1.3, 3)
        for (s_x, s_y, s_w, s_h) in smiles:
            cv2.rectangle(face_frame_color, (s_x, s_y), (s_x + s_w, s_y + s_h), (0, 255, 0), 2)
    return original_frame

video_capture = cv2.VideoCapture(0)
while True:
    original_frame = video_capture.read()[1]
    canvas = detect(original_frame)
    cv2.imshow("Detecting smiles", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
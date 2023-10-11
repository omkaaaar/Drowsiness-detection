import cv2
from playsound import playsound

# Load the pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Parameters for drowsiness detection
closed_eye_frames = 0  # Number of consecutive frames with closed eyes
drowsy_threshold = 15  # Adjust as needed

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for better eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the detected face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        if len(eyes) == 0:
            closed_eye_frames += 1
            if closed_eye_frames >= drowsy_threshold:
                print("Drowsiness detected. Beep sound will be played.")
                playsound("beep.mp3")  # You can use any sound file you prefer
        else:
            closed_eye_frames = 0

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

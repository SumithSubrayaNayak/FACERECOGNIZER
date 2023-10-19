import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the image file paths relative to the script's directory
sumith_image_path = os.path.join(script_dir, "faces", "sumith.jpg")
rohan_image_path = os.path.join(script_dir, "faces", "rohan.jpg")

video_capture = cv2.VideoCapture(0)

# Load known faces
sumiths_image = face_recognition.load_image_file(sumith_image_path)
sumith_encoding = face_recognition.face_encodings(sumiths_image)[0]

rohans_image = face_recognition.load_image_file(rohan_image_path)
rohan_encoding = face_recognition.face_encodings(rohans_image)[0]

known_face_encodings = [sumith_encoding, rohan_encoding]
known_face_names = ["sumith", "rohan"]

# List of expected students
students = known_face_names.copy()
face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        # Proceed with resizing and other operations
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    else:
        print("Invalid frame")

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Add the text if a person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                linetype = 2

                cv2.putText(frame, name + " present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, linetype)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

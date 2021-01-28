import os
import face_recognition
import cv2
import dlib
import numpy as np

json_dic={"face_detected":False,"Name":None,"Attentive":None}
known_face_encodings = []
known_face_names = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

K = [6.2500000000000000e+002, 0.0, 3.1250000000000000e+002,
     0.0, 6.2500000000000000e+002, 3.1250000000000000e+002,
     0.0, 0.0, 1.0]

model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left cornerq
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

for dir , _ ,files in os.walk("labeled images"):
    for image in files:
        image_data=face_recognition.load_image_file("labeled images/"+image)
        image_encoding=face_recognition.face_encodings(image_data)[0]
        known_face_encodings.append(image_encoding)
        known_face_names.append(image[:-4])


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    size = frame.shape #

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #

    faces = detector(gray)  # ; print(faces)
    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        json_dic["face_detected"]=False if(len(face_locations)==0) else True
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        faces = detector(gray)  # ; print(faces)
        for face in faces:
            # The face landmarks code begins from here
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            landmarks = predictor(gray, face)

            # We are then accesing the landmark points
            i = [33, 8, 36, 45, 48,
                 54]  # Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, right mouth corner
            image_points = []
            for n in i:
                x = landmarks.part(n).x
                y = landmarks.part(n).y;
                # image_points = np.array([(x,y)], dtype="double")
                image_points += [(x, y)]
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

            image_points = np.array(image_points, dtype="double")
            # print(image_points)
            # print("Camera Matrix :\n {0}".format(cam_matrix))

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                          cam_matrix, dist_coeffs,
                                                                          flags=cv2.SOLVEPNP_ITERATIVE)

            # print("Rotation Vector:\n {0}".format(rotation_vector))
            # print("Translation Vector:\n {0}".format(translation_vector))

            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector,
                                                             cam_matrix, dist_coeffs)

            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

           # print("distance : " + str(np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))) + "  p1 : " + str(
            #    p1) + "  p2 : " + str(p2))
            radius=np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
            json_dic["Attentive"]=True if radius <80 else False
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
       # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        json_dic["Name"]=name

    # Display the resulting image
    cv2.imshow('Video', frame)
    print(json_dic)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

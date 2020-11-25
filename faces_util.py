import glob
import os
import pickle

import cv2
import face_recognition
import numpy as np


class FacesError(Exception):
    def __init__(self, error_info):
        super().__init__(self)
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class Faces(object):
    FACES_DATA_FILE = 'faces.data'

    def __init__(self, face_dir=None):
        self.faces_encodings = []
        self.faces_names = []
        self.face_dir = face_dir
        self.faces_data_init()

    def _load_faces_from_folder(self):
        if self.face_dir is not None:
            path = os.path.join(self.face_dir)
            list_of_files = [f for f in glob.glob(path + '/*.jpg')]
            return list_of_files

    def faces_encoding_from_files(self, img_files):
        for img_file in img_files:
            face_file = face_recognition.load_image_file(img_file)
            # face encoding
            face_encodings = face_recognition.face_encodings(face_file)
            if len(face_encodings) > 0: # check if a face exists in an image
                self.faces_encodings.append(face_encodings[0])
                # face name
                face_name = os.path.basename(img_file).replace('.jpg', '')
                self.faces_names.append(face_name)
            else:
                print("the image file({}) does not include face!!!".format(img_file))

    def faces_encoding_from_folder(self):
        list_of_files = self._load_faces_from_folder()
        self.faces_encoding_from_files(list_of_files)

    def faces_serialization(self):
        """
            save the face encodings and names into the local file
        """
        with open(Faces.FACES_DATA_FILE, 'wb') as f:
            pickle.dump((self.faces_names, self.faces_encodings), f)

    def faces_deserialization(self):
        """
            load the face encodings and names from the local file
        """
        with open(Faces.FACES_DATA_FILE, 'rb') as f:
            self.faces_names, self.faces_encodings = pickle.load(f)

    def faces_data_init(self):
        """
            the priority data source with initialization:
            1.local face library data
            2.the given image dir
            3.otherwise, error
        """
        if os.path.exists(Faces.FACES_DATA_FILE):
            print("Faces init from faces lib。")
            self.faces_deserialization()
        elif self.face_dir is not None:
            print("Faces init from faces folder, need faces images training, please wait。")
            self.faces_encoding_from_folder()
            self.faces_serialization()
        else:
            raise FacesError("faces data init failed!")

        if len(self.faces_encodings) == 0:
            raise FacesError("No faces data detected!")

    def get_faces_info(self):
        return self.faces_names

    def faces_process(self, frame):
        """
            face detection and recognition in the image frame which are from camera
        """
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # face detection
        cam_face_locations = face_recognition.face_locations(rgb_small_frame)
        cam_face_encodings = face_recognition.face_encodings(rgb_small_frame, cam_face_locations)

        # face recognition and compare the results to the face library
        face_names_camera = []
        for cam_face_encoding in cam_face_encodings:
            matches = face_recognition.compare_faces(self.faces_encodings, cam_face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.faces_encodings, cam_face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.faces_names[best_match_index]
            face_names_camera.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(cam_face_locations, face_names_camera):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Input text label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame


def cam_face_recognition(faces):
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = video_capture.read()
        frame = faces.faces_process(frame)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    DIR = 'C:\\Users\\test\\PycharmProjects\\python-gui\\pyqt\\examples-_\\src\\face\\faces'
    faces = Faces(DIR)
    cam_face_recognition(faces)

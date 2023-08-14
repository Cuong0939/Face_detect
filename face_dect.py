# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe import solutions

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success,image = cap.read()
        if not success:
            print('Ignoring empty camera frame.')
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(image)

        image.flags.writeable = True
        image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_contours_style())   
                
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image,1))
        print(image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
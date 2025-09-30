import cv2
import mediapipe as mp

#Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8
)

image = cv2.imread("test.jpg")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

#Facial Landmarks
result = face_mesh.process(image)

#Image and Image proccessing
screen_width = 800
screen_height = 600
scale = min(screen_width / width, screen_height / height)

if scale < 1:
    adjusted_width = int(width * scale)
    adjusted_height = int(height * scale)
    image = cv2.resize(image, (adjusted_width, adjusted_height), interpolation=cv2.INTER_AREA)

for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, 468):
        pt = facial_landmarks.landmark[i]
        x = int(pt.x * adjusted_width)
        y = int(pt.y * adjusted_height)

        cv2.circle(image, (x, y), 3, (100, 0, 0), -1)

#Image display
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import face_recognition

print('Running Face Recognition .....')

# Load Two images from specified path and convert to RGB format (for computation RGB format used)
imgVK1 = face_recognition.load_image_file('images/vk1.jpg')           # give path/filename bw ' '
imgVK1 = cv2.cvtColor(imgVK1, cv2.COLOR_BGR2RGB)
imgVK2 = face_recognition.load_image_file('images/mtn.jpg')         # give path/filename bw ' '
imgVK2 = cv2.cvtColor(imgVK2, cv2.COLOR_BGR2RGB)

# use the face recognition package's functions to identify the face position and store it
# encode the features and store
# draw a box around the detected face region
try:
    faceLocVK1 = face_recognition.face_locations(imgVK1)[0]
except IndexError as a:
    print('No Face Detected in image 1')
    exit(0)
encodeVK1 = face_recognition.face_encodings(imgVK1)[0]
cv2.rectangle(imgVK1, (faceLocVK1[3], faceLocVK1[0]), (faceLocVK1[1], faceLocVK1[2]), (255, 255, 0), 2)

try:
    faceLocVK2 = face_recognition.face_locations(imgVK2)[0]
except IndexError as a:
    print('No Face Detected in image 2')
    exit(0)
encodeVK2 = face_recognition.face_encodings(imgVK2)[0]
cv2.rectangle(imgVK2, (faceLocVK2[3], faceLocVK2[0]), (faceLocVK2[1], faceLocVK2[2]), (255, 255, 0), 2)

# compare the two encoded values and store the result
# a list with [True] or [False] is returned depending on the result
result = face_recognition.compare_faces([encodeVK1], encodeVK2)

# Print the result
if (result == [True]):
    print('Faces Match')
else:
    print("Faces not matched")

# display the two images
cv2.imshow('Image 1', imgVK1)
cv2.imshow('Image 2', imgVK2)
cv2.waitKey(0)



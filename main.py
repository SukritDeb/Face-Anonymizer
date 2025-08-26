import os
import argparse

import cv2


def process_img(img, face_detector):

    H, W, _ = img.shape

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # blur faces
        img[y:y + h, x:x + w, :] = cv2.blur(img[y:y + h, x:x + w, :], (30, 30))

    return img


args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)

args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# detect faces (Haar cascade instead of mediapipe)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if args.mode in ["image"]:
    # read image
    img = cv2.imread(args.filePath)

    img = process_img(img, face_detector)

    # save image
    cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

elif args.mode in ['video']:

    cap = cv2.VideoCapture(args.filePath)
    ret, frame = cap.read()

    output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                   cv2.VideoWriter_fourcc(*'MP4V'),
                                   25,
                                   (frame.shape[1], frame.shape[0]))

    while ret:

        frame = process_img(frame, face_detector)

        output_video.write(frame)

        ret, frame = cap.read()

    cap.release()
    output_video.release()

elif args.mode in ['webcam']:
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    while ret:
        frame = process_img(frame, face_detector)

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # press q to exit
            break

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


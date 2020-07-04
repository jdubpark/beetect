import argparse
import cv2
import os


parser = argparse.ArgumentParser(description='Data Frame Generator')

parser.add_argument('--video', type=str, help='path to video file')
parser.add_argument('--dest', type=str, help='image output destination')


if __name__ == '__main__':
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    frame_count = 0
    ret = True
    while ret is True:
        ret, frame = cap.read()
        if ret is False:
            break
        image_path = os.path.join(args.dest, '{0}.jpg'.format(frame_count))
        cv2.imwrite(image_path, frame)
        frame_count += 1

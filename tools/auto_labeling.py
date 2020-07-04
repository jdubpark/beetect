import argparse
import xml.etree.cElementTree as ET

import torch

from beetect.modeling import resnet50_fpn
from beetect.inference import run_inference


parser = argparse.ArgumentParser(description='Auto labeling with a trained model')

parser.add_argument('-i', '--image', type=str, dest='image',
                    help='path to images file (mutually exclusive to video)')
parser.add_argument('-v', '--video', type=str, dest='video',
                    help='path to video file (mutually exclusive to image)')
parser.add_argument('-c', '--checkpoint', type=str, dest='checkpoint',
                    help='path to checkpoint file')
parser.add_argument('-o', '--output', type=str, dest='output',
                    help='label file output directory')
parser.add_argument('--iou', type=float, default=0.2,
                    help='IoU threshold (default 0.2)')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.video is None:
        raise ValueError('Invalid path: video')
    if args.checkpoint is None:
        raise ValueError('Invalid path: checkpoint')

    model = resnet50_fpn()
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])
    arch = checkpoint['arch']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(arch, epoch, loss)
    print('=' * 10)

    outputs = [] # index ordered frames

    if args.image:
        image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = run_inference(image, model, device, iou_threshold=args.iou)
        outputs.append(output)

    elif args.video:
        cap = cv2.VideoCapture(args.video)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = run_inference(frame, model, device, iou_threshold=args.iou)
            outputs.append(output)

        cap.release()

    # CVAT 1.1 XML format
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'

    # meta
    meta = ET.SubElement(root, 'meta')
    task = ET.SubElement(meta, 'task')
    labels = ET.SubElement(task, 'labels')
    label1 = ET.SubElement(labels, 'label')

    # label
    ET.SubElement(label1, 'name').text = 'Bee'
    l1_attr = ET.SubElement(ET.SubElement(label1, 'attributes'), 'attribute')
    ET.SubElement(l1_attr, 'name').text = 'Shape'
    ET.SubElement(l1_attr, 'mutable').text = True
    ET.SubElement(l1_attr, 'input_type').text = 'radio'
    ET.SubElement(l1_attr, 'default_value').text = 'Body'
    ET.SubElement(l1_attr, 'input_type').text = 'Body\nHead\nButt'

    # tracks


    outputs

    # finish up
    tree = ET.ElementTree(root)
    tree.write('test-auto.xml')

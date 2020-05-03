import xml.etree.ElementTree as ET
import urllib.request
import csv
import mimetypes
import random
from pathlib import Path

import requests
import cv2

from beetect.config import custom as config


def main():
    C = config.get(Path.cwd())

    img_filepaths = get_images(C.dir.images)
    print(img_filepaths)


def list_dir(dir, ext=''):
    if ext == '':
        # grab filenames with all ext
        list = [{'name': f.name, 'stem': f.stem} for f in Path(dir).iterdir() if f.is_file() and f.name[0] != '.']
    elif ext == 'stem' or ext == 'name':
        # special 'stem'/'name', return all filenames (without/with ext)
        list = [f.stem if ext == 'stem' else f.name for f in Path(dir).iterdir() if f.is_file() and f.name[0] != '.']
    else:
        # grab filenames with specific ext (e.stem returns filename without ext)
        list = [{'name': f.name, 'stem': f.stem} for f in Path(dir).rglob('*.'+ext)]

    return list


def get_synset(wnid, synset_dir):
    """ Wrap around get_synset_csv() to return dict of wnid_name: url """
    lines = get_synset_csv(wnid, synset_dir)

    # convert to dictionary, with key 'wnid name' and val 'img url'
    data = {x[0]: x[1] for x in lines}

    return data


def get_synset_csv(wnid, synset_dir):
    """ Get synset mapped urls (in csv format) for bbox annotations """
    filepath = (synset_dir / wnid).with_suffix('.csv')
    # print(filepath)

    if Path.exists(filepath):
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            data = list(reader)

        return data

    else:
        url = 'http://image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid='+wnid
        lines = [x.decode('utf-8').rstrip().split(' ') for x in urllib.request.urlopen(url) if x != '']
        # remove empty sub list (filter returns iter)
        lines = list(filter(lambda x: x[0], lines))

        with open(filepath, 'w', newline='') as f:
            wr = csv.writer(f)
            for line in lines:
                wr.writerow(line) # row of [wnid_cat_name, img_url]

        return lines


def get_xml(filename, annot_dir):
    """
    Return xml root of a given file in pre-set annot dir
    input: filename -> stem (without ext)
    """
    # dir = Path(annot_dir)
    filepath = (annot_dir / filename).with_suffix('.xml')
    tree = ET.parse(filepath)
    root = tree.getroot()
    return root


def xml_pascal_bbox(root):
    """ ImageNet bbox annotation format is Pascal VOC """
    fname = root.find('filename')
    obj = root.find('object')
    bbox = obj.find('bndbox')
    # get bounding box values (child.tag == x y name, child.text == value)
    xys = [int(child.text) for child in bbox]
    # ((x min, y min), (x max, y max))
    return ((xys[0], xys[1]), (xys[2], xys[3]))


def retrieve_image(file_url, filename, img_dir):
    """
    Retreieve (download) image based on passed synset wnid
    input: filename -> filename stem in {name, stem}
    """
    img_filenames = list_dir(img_dir)

    # print(filename)
    # skip download and return dir (with ext) if image is already downloaded
    for img_filename in img_filenames:
        if filename == img_filename['stem']:
            return img_dir / img_filename['name']

    # try:
    #     res = requests.get(file_url, timeout=10)
    #     res.raise_for_status()
    # except requests.exceptions.HTTPError as err:
    #     # raise SystemExit(err)
    #     return None
    try:
        res = requests.get(file_url, timeout=5)
    except requests.ConnectionError:
        return None

    if not res.ok: return None

    content_type = res.headers['content-type']
    ext = mimetypes.guess_extension(content_type) # get ext for img name ext
    # print(file_url, 'ext', ext)
    if not ext: return None

    img_filepath = (img_dir / filename).with_suffix(ext)

    with open(img_filepath, 'wb') as f:
        f.write(res.content)

    return img_filepath


def get_image_data(filename, wnid, annot_dir, synset_dir):
    """ Return bounding box data and file url for a given filename (wnid) """
    xml = get_xml(filename['stem'], annot_dir)
    bbox = xml_pascal_bbox(xml)
    # print(bbox)

    synset = get_synset(wnid, synset_dir)
    fn_stem = filename['stem']
    file_url = synset[fn_stem]

    return bbox, file_url


def get_images(img_dir):
    """ Get all images, no downloading """
    img_filepaths = [f for f in Path(img_dir).iterdir() if f.is_file() and f.name[0] != '.']
    return list(filter(None, img_filepaths))


def download_images(wnid, annot_dir, img_dir, synset_dir):
    """ Download all images (in bbox annot dir) and return all of their filepaths """
    bbox_filenames = list_dir(annot_dir, 'xml')
    img_filepaths = []

    for filename in bbox_filenames:
        bbox, file_url = get_image_data(filename, wnid, annot_dir, synset_dir)

        img_filepath = retrieve_image(file_url, filename['stem'], img_dir)
        # if not img_filepath: continue # skip if image can't be downloaded/retrieved
        # # convert PosixPath to str (for cv2)
        # img_filepath = img_filepath.absolute().as_posix()
        img_filepaths.append(img_filepath)

    return img_filepaths


if __name__ == '__main__':
    main()

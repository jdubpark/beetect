from pathlib import Path

from beetect.utils.dict_map import Map

def get(cwd):
    conf = Map({})

    if not cwd:
        cwd = Path.cwd()

    # word net
    conf.wnid = 'n02206856'
    conf.wnname = conf.wnid+'-bee'

    # directory
    conf.dir = Map({})
    conf.dir.dataset = cwd / 'dataset' / 'image_net'
    conf.dir.annots = conf.dir.dataset / 'bbox'
    conf.dir.images = conf.dir.dataset / 'images'
    conf.dir.synset = conf.dir.dataset

    return conf

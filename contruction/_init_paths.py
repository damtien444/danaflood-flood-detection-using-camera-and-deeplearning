import os.path as osp
import sys


def add_path(res_path):
    if res_path not in sys.path:
        sys.path.insert(0, res_path)


this_dir = osp.dirname(__file__)

add_path(osp.join(this_dir))

path = osp.join(this_dir, 'api/')
add_path(path)

path = osp.join(this_dir, 'code/')
add_path(path)

path = osp.join(this_dir, 'model_zoo/')
add_path(path)

path = osp.join(this_dir, 'website/')
add_path(path)


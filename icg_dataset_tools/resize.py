import sys

from os import walk
from PIL import Image

img_dir = sys.argv[1]
out_dir = sys.argv[2]

files = []
for (dirpath, dirnames, filenames) in walk(img_dir):
    files.extend(filenames)
    break
if not files:
    print('no images found in input directory')
    exit()

for f in files:
    print('resizing %s...' % f)
    with Image.open('%s/%s' % (img_dir, f)) as image:
        im_resize = image.resize((500, 333))
        im_resize.save('%s/%s' % (out_dir, f))

import os
import json
from PIL import Image
import glob

DATA_PATH = '/home/user-1/Detection/tracking/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['val', 'train']
ANNOTATIONS_FOLDER = DATA_PATH + 'annotations/'

def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records

def process_annotation_file(ann_file, out, split):
    anns_data = load_func(ann_file)
    image_cnt = len(out['images'])
    ann_cnt = len(out['annotations'])

    for ann_data in anns_data:
        image_cnt += 1
        file_path = DATA_PATH + 'PigDataset_{}/'.format(split) + '{}.jpg'.format(ann_data['ID'])
        im = Image.open(file_path)
        image_info = {'file_name': '{}.jpg'.format(ann_data['ID']), 
                      'id': image_cnt,
                      'height': im.size[1], 
                      'width': im.size[0]}
        out['images'].append(image_info)

        if split != 'test':
            anns = ann_data['gtboxes']
            for i in range(len(anns)):
                ann_cnt += 1
                fbox = anns[i]['fbox']
                ann = {'id': ann_cnt,
                       'category_id': 1,
                       'image_id': image_cnt,
                       'track_id': -1,
                       'bbox_vis': anns[i]['vbox'],
                       'bbox': fbox,
                       'area': fbox[2] * fbox[3],
                       'iscrowd': 1 if 'extra' in anns[i] and \
                                       'ignore' in anns[i]['extra'] and \
                                       anns[i]['extra']['ignore'] == 1 else 0}
                out['annotations'].append(ann)

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    for split in SPLITS:
        out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'pig'}]}
        # List all annotation files for the current split
        annotation_files = glob.glob(os.path.join(ANNOTATIONS_FOLDER, f'*_{split}.odgt'))

        for ann_file in annotation_files:
            process_annotation_file(ann_file, out, split)

        out_path = OUT_PATH + f'{split}.json'
        print(f'Writing aggregated data for {split} to {out_path}')
        json.dump(out, open(out_path, 'w'))

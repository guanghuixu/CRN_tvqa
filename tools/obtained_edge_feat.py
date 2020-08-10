import numpy as np
import os
import operator
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser("Process Dataset")
parser.add_argument("--dataset", type=str, default="textvqa",
                    help="textvqa, stvqa, ocrvqa")
parser.add_argument("--imdb_path", type=str, default="data/textvqa/imdb/imdb_val_ocr_en.npy",)
args = parser.parse_args()

edge_path = 'data/{}/edge_feat'.format(args.dataset)
if not os.path.exists(edge_path):
    os.makedirs(edge_path)

imdb = np.load(args.imdb_path, allow_pickle=True).tolist()
# dict_keys(['image_name', 'image_path', 'image_width', 'image_height', 'feature_path', 
# 'image_id', 'question', 'question_id', 'question_tokens', 'valid_answers', 'answers', 
# 'ocr_tokens', 'ocr_info', 'ocr_normalized_boxes', 'obj_normalized_boxes'])


for fea in tqdm(imdb[1:]):
    image_path = fea['image_path']
    filepath, tempfilename = os.path.split(image_path)
    filename, extension = os.path.splitext(tempfilename)
    save_path = os.path.join(edge_path, filepath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    info_path = os.path.join(save_path, filename+'_info.npy')
    if os.path.exists(info_path):
        continue
    feat_path = os.path.join(save_path, filename+'.npy')

    fea_ = {}
    obj_ocr_fea = np.zeros((100, 50, 5))
    ocr_obj_fea = np.zeros((50, 100, 5))

    obj_bbox = fea['obj_normalized_boxes']
    ocr_bbox = fea['ocr_normalized_boxes']
    img_h = fea['image_height']
    img_w = fea['image_width']

    obj_bbox[:, 0] = obj_bbox[:, 0] * img_w
    obj_bbox[:, 1] = obj_bbox[:, 1] * img_h
    obj_bbox[:, 2] = obj_bbox[:, 2] * img_w
    obj_bbox[:, 3] = obj_bbox[:, 3] * img_h

    ocr_bbox[:, 0] = ocr_bbox[:, 0] * img_w
    ocr_bbox[:, 1] = ocr_bbox[:, 1] * img_h
    ocr_bbox[:, 2] = ocr_bbox[:, 2] * img_w
    ocr_bbox[:, 3] = ocr_bbox[:, 3] * img_h

    for m, obj_box_m in enumerate(obj_bbox):
        rcx, rcy, rw, rh = (obj_box_m[0] + obj_box_m[2] + 1)/2, (obj_box_m[1] + obj_box_m[3] + 1)/2, (obj_box_m[2] - obj_box_m[0] + 1), (obj_box_m[3] - obj_box_m[1] + 1)
        for n, ocr_box_n in enumerate(ocr_bbox):
            if n >= 50:
                break
            obj_ocr_edge_feats = np.array([
                (ocr_box_n[0] - rcx) / rw,
                (ocr_box_n[1] - rcy) / rh,
                (ocr_box_n[2] - rcx) / rw,
                (ocr_box_n[3] - rcy) / rh,
                ((ocr_box_n[2] - ocr_box_n[0] + 1) * (ocr_box_n[3] - ocr_box_n[1] + 1)) / (rw * rh)
            ])
            obj_ocr_fea[m, n:(n+1), :5] = obj_ocr_edge_feats
    
    for a, ocr_box_a in enumerate(ocr_bbox):
        rcx, rcy, rw, rh = (ocr_box_a[0] + ocr_box_a[2] + 1)/2, (ocr_box_a[1] + ocr_box_a[3] + 1)/2, (ocr_box_a[2] - ocr_box_a[0] + 1), (ocr_box_a[3] - ocr_box_a[1] + 1)
        if a >= 50:
            break
        for b, obj_box_b in enumerate(obj_bbox):
            ocr_obj_edge_feats = np.array([
                (obj_box_b[0] - rcx) / rw,
                (obj_box_b[1] - rcy) / rh,
                (obj_box_b[2] - rcx) / rw,
                (obj_box_b[3] - rcy) / rh,
                ((obj_box_b[2] - obj_box_b[0] + 1) * (obj_box_b[3] - obj_box_b[1] + 1)) / (rw * rh)
            ])
            ocr_obj_fea[a, b:(b+1), :5] = ocr_obj_edge_feats

    fea_['obj_ocr_edge_feat'] = obj_ocr_fea
    fea_['ocr_obj_edge_feat'] = ocr_obj_fea
    np.save(info_path, fea_, allow_pickle=True)
    np.save(feat_path, np.array([], dtype=np.float32).reshape([0, 4]), allow_pickle=True)
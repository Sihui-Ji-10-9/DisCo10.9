import torch
from collections import defaultdict
import sys
import os
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
from utils.common import encoded_from_img
from utils.tsv_io import tsv_writer
from PIL import Image
import json, cv2, math, yaml
import numpy as np
import argparse
from joblib import load
from torchvision import transforms
tensor_transforms = transforms.Compose(
    [
        transforms.Normalize([0.5], [0.5]),
    ]
)
def write_to_yaml_file(context, file_name):
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, encoding='utf-8')

def load_image(path):
    if os.path.exists(path):
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
    else:
        image = None
    return image

def load_video_paths(args):
    folder_path = args.root_folder
    vid_list = os.listdir(folder_path)
    video_paths = [ folder_path + '/' + vid_name for vid_name in vid_list]
    video_paths.sort()
    return video_paths

def load_frame_paths(video_folder):
    frm_list = os.listdir(video_folder)
    frame_paths = [ video_folder + '/' + frm_name for frm_name in frm_list if 'png' in frm_name or 'jpg' in frm_name]
    frame_paths.sort()
    return frame_paths

def get_frm_list(args):
    video_paths = load_video_paths(args)
    all_frm_paths = []
    for vid in video_paths:
        all_frm_paths += load_frame_paths(vid)
    return all_frm_paths

def load_image_paths(image_folder):
    frm_list = os.listdir(image_folder)
    frame_paths = [ image_folder + '/' + frm_name for frm_name in frm_list if frm_name.endswith(".png") or frm_name.endswith(".jpg")]
    frame_paths.sort()
    return frame_paths

def load_pkl_paths(image_folder):
    frm_list = os.listdir(image_folder)
    frame_paths = [ image_folder + '/' + frm_name for frm_name in frm_list if frm_name.endswith(".pkl")]
    frame_paths.sort()
    return frame_paths

def load_npy_paths(image_folder):
    frm_list = os.listdir(image_folder)
    frame_paths = [ image_folder + '/' + frm_name for frm_name in frm_list if frm_name.endswith(".npy")]
    frame_paths.sort()
    return frame_paths

def main(args):
    split = args.split 
    
    image_path_list = load_image_paths(os.path.join(args.root_folder,'image2'))
    cloth_path_list = load_image_paths(os.path.join(args.root_folder,'cloth2'))
    image_mask_path_list = load_image_paths(os.path.join(args.root_folder,'image-parse-v3-2'))
    cloth_mask_path_list = load_image_paths(os.path.join(args.root_folder,'cloth-mask2'))
    # smpl_img_path_list = load_image_paths(os.path.join(args.root_folder,'smpl_new'))
    # smpl_pkl_path_list = load_pkl_paths(os.path.join(args.root_folder,'smpl'))
    # densepose_img_path_list = load_image_paths(os.path.join(args.root_folder,'image-densepose'))

    # if args.debug:
    #     image_path_list = image_path_list[:10]

    ###############################################################################
    # process img.tsv
    if args.img:
        print('generating img.tsv')
    def gen_row(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            image = load_image(image_path)
            if image is None:
                print(f"image does not exists: {image_path}")
                continue
            row = [image_name, encoded_from_img(image)]
            yield(row)
    if args.img:
        tsv_writer(gen_row(image_path_list), f"{args.output_folder}/{split}_img.tsv")
    ###############################################################################
    # process cloth.tsv
    if args.cloth:
        print('generating cloth.tsv')
    def gen_row(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            image = load_image(image_path)
            if image is None:
                print(f"image does not exists: {image_path}")
                continue
            row = [image_name, encoded_from_img(image)]
            yield(row)
    if args.cloth:
        tsv_writer(gen_row(cloth_path_list), f"{args.output_folder}/{split}_cloth.tsv")

    ###############################################################################
    # process smpl.tsv
    if args.smpl:
        print('generating smpl.tsv')
    def gen_row(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            image = load_image(image_path)
            if image is None:
                print(f"image does not exists: {image_path}")
                continue
            row = [image_name, encoded_from_img(image)]
            yield(row)
    if args.smpl:
        tsv_writer(gen_row(smpl_img_path_list), f"{args.output_folder}/{split}_smpl_new.tsv")

    ###############################################################################
    # process densepose.tsv
    if args.densepose:
        print('generating densepose.tsv')
    def gen_row(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            # print(image_name)
            image = load_image(image_path)
            if image is None:
                print(f"image does not exists: {image_path}")
                continue
            row = [image_name, encoded_from_img(image)]
            yield(row)
    if args.densepose:
        tsv_writer(gen_row(densepose_img_path_list), f"{args.output_folder}/{split}_densepose.tsv")
    ###############################################################################
    
    # process caption.tsv
    if args.caption:
        print('generating caption.tsv')
    def gen_vis2line_row(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            # image = load_image(f"./tiktok_datasets/{image_path}")
            # if image is None:
            #     print(f"image does not exists: {image_path}")
            #     continue
            row = [image_name, ' ']
            yield(row)
    if args.caption:
        tsv_writer(gen_vis2line_row(image_path_list), f"{args.output_folder}/{split}_caption.tsv")
    '''
    ###############################################################################
    # process pose.tsv
    if args.pose:
        print('generating poses.tsv')
    # draw the body keypoint and lims
    def draw_bodypose(canvas, coco_keypoints):
        pose = [
                coco_keypoints[0], # Nose (0)
                list((np.asarray(coco_keypoints[5]) + np.asarray(coco_keypoints[6]))/2), # Neck (1)
                coco_keypoints[6], # RShoulder (2)
                coco_keypoints[8], # RElbow (3)
                coco_keypoints[10], # RWrist (4)
                coco_keypoints[5], # LShoulder (5)
                coco_keypoints[7], # LElbow (6)
                coco_keypoints[9], # LWrist (7)
                coco_keypoints[12], # RHip (8)
                coco_keypoints[14], # RKnee (9)
                coco_keypoints[16], # RAnkle (10)
                coco_keypoints[11], # LHip (11)
                coco_keypoints[13], # LKnee (12)
                coco_keypoints[15], # LAnkle (13)
                coco_keypoints[2], # REye (14)
                coco_keypoints[1], # LEye (15)
                coco_keypoints[4], # REar (16)
                coco_keypoints[3], # LEar (17)
            ]  # openpose_keypoints

        stickwidth = 4
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        canvas = np.zeros_like(canvas)

        for i in range(18):
            x, y = pose[i][0:2]
            if x>=0 and y>=0: 
                cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
                # cv2.putText(canvas, '%d'%(i), (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        for limb_idx in range(17):
            cur_canvas = canvas.copy()
            index_a = limbSeq[limb_idx][0]-1
            index_b = limbSeq[limb_idx][1]-1

            if pose[index_a][0]<0 or pose[index_b][0]<0 or pose[index_a][1]<0 or pose[index_b][1]<0:
                continue

            Y = [pose[index_a][0], pose[index_b][0]]
            X = [pose[index_a][1], pose[index_b][1]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[limb_idx])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        # Convert color space from BGR to RGB
        # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        # Create PIL image object from numpy array
        canvas = Image.fromarray(canvas)
        return canvas

    def gen_row_pose(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            video_key = image_path.split("/")[-2]
            image_key = f"Custom_{video_key}_{image_name}"
            image = load_image(image_path)
            if image is None:
                print(f"image does not exists: {image_path}")
                continue
            # anno_pose_path = "./tiktok_datasets/" + image_path.replace("/images/", "/openpose_json/")+".json"
            anno_pose_path = "{}/openpose_json/{}.json".format(('/').join(image_path.split("/")[0:-1]),image_name)
            pose_without_visibletag = []
            f = open(anno_pose_path,'r')
            d = json.load(f)
            # if there is a valid openpose skeleton, load it
            if len(d)>0:
                for j in range(17):
                    x = d[0]['keypoints'][j][0]
                    y = d[0]['keypoints'][j][1]
                    pose_without_visibletag.append([x,y])
            else: # if there is not valid openpose skeleton, add a dummy one
                for j in range(17):
                    x = -1
                    y = -1
                    pose_without_visibletag.append([x,y])      
            pose_image = draw_bodypose(image, pose_without_visibletag)
            row = [image_key, encoded_from_img(pose_image)]
            yield(row)
    if args.pose:
        tsv_writer(gen_row_pose(image_path_list), f"{args.output_folder}/{split}_poses.tsv")
    '''
    ###############################################################################
    # process mask.tsv
    if args.cloth_mask:
        print('generating cloth_mask.tsv')
    def gen_row_mask(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            try:
                if os.path.exists(image_path):
                    mask_image = load_image(image_path)
                    # img = np.asarray(img)
                    # bk = img[:,:,:]==[68,0,83]
                    # fg = (bk==False)
                    # fg = fg*255.0
                    # mask = fg.astype(np.uint8)
                    # ipl_mask = Image.fromarray(mask)
                else:
                    mask_image = None
            except Exception as e: 
                print(e)
                mask_image = None
            valid = True
            if mask_image is None:
                valid = False
            row = [image_name, encoded_from_img(mask_image), valid]
            yield(row)
    if args.cloth_mask:
        tsv_writer(gen_row_mask(cloth_mask_path_list), f"{args.output_folder}/{split}_cloth_mask.tsv")
    ###############################################################################
    # process img_mask.tsv
    if args.img_mask:
        print('generating img_mask.tsv')

    def gen_row_mask(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            try:
                if os.path.exists(image_path):
                    img = load_image(image_path)
                    gray_image = img.convert("L")           
                    # 设定阈值
                    threshold = 1
                    # 进行二值化处理
                    mask_image = gray_image.point(lambda pixel: 0 if pixel < threshold else 255, "1")
                    mask_image = mask_image.convert("RGB")
                    # bk = img[:,:,:]==[68,0,83]
                    # fg = (bk==False)
                    # fg = fg*255.0
                    # mask = fg.astype(np.uint8)
                else:
                    mask_image = None

            except Exception as e: 
                print(e)
                mask_image = None
            valid = True
            if mask_image is None:
                valid = False
            row = [image_name, encoded_from_img(mask_image), valid]
            yield(row)
    if args.img_mask:
        tsv_writer(gen_row_mask(image_mask_path_list), f"{args.output_folder}/{split}_img_mask.tsv")
    
    ###############################################################################
    
    # process shape.tsv
    if args.shape:
        print('generating shape.tsv')
    
    def gen_vis2line_row(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            # image = load_image(f"./tiktok_datasets/{image_path}")
            # if image is None:
            #     print(f"image does not exists: {image_path}")
            #     continue
            output = load(image_path)
            shape = list(output['pred_shape'][0:1].squeeze())
            row = [image_name, shape]
            yield(row)
    if args.shape:
        tsv_writer(gen_vis2line_row(smpl_pkl_path_list), f"{args.output_folder}/{split}_shape.tsv")
    
    ###############################################################################
    '''
    # process densepose.tsv
    if args.densepose:
        print('generating densepose.tsv')
    
    def gen_vis2line_row(image_path_list):
        for i, image_path in enumerate(image_path_list):
            if i % 100 == 0:
                print(f"{i}/{len(image_path_list)}")
            image_name = image_path.split("/")[-1]
            # image = load_image(f"./tiktok_datasets/{image_path}")
            # if image is None:
            #     print(f"image does not exists: {image_path}")
            #     continue
            dp = torch.from_numpy(np.load(image_path).astype('float16')).unsqueeze(0)
            trans_dp=tensor_transforms(dp)
            # npdp=np.array(trans_dp)
            # dp = list(trans_dp)
            # print(npdp.shape)
            # dp2np = np.array(dp)
            # print(dp2np.shape)
            row = [image_name, trans_dp]
            yield(row)
    if args.densepose:
        tsv_writer(gen_vis2line_row(npy_path_list), f"{args.output_folder}/{split}_densepose.tsv")
    '''
    ###############################################################################
    # process yaml file
    if args.yaml:
        all_field = ['img','img_mask','shape','smpl','cloth','cloth_mask','caption','img_densepose']
        out_cfg = {'composite': False}

        for field in all_field:
            tsvfilename = f'{args.split}_{field}.tsv'
            out_cfg[field] = tsvfilename

        write_to_yaml_file(out_cfg, os.path.join(args.output_folder, args.split + '.yaml'))

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder",
                        type=str, default='./HOME/HOME/jisihui/VITON/test')
    parser.add_argument("--output_folder",
                        type=str, default='./blob_dir/debug_output/video_sythesis/dataset')
    parser.add_argument("--split",
                        type=str, default='train')
    parser.add_argument("--debug",
                        type=bool, default=False)
    parser.add_argument('--cloth',  type=str_to_bool,
                            nargs='?', const=True, default=False)
    parser.add_argument('--cloth_mask',  type=str_to_bool,
                            nargs='?', const=True, default=False)
    parser.add_argument('--img',  type=str_to_bool,
                            nargs='?', const=True, default=False)
    parser.add_argument('--img_mask',  type=str_to_bool,
                            nargs='?', const=True, default=False)
    parser.add_argument('--shape',  type=str_to_bool,
                            nargs='?', const=True, default=False)
    parser.add_argument('--smpl',  type=str_to_bool,
                            nargs='?', const=True, default=False)
    parser.add_argument('--caption',  type=str_to_bool,
                            nargs='?', const=True, default=False)
    parser.add_argument('--densepose',  type=str_to_bool,
                            nargs='?', const=True, default=False)
    parser.add_argument('--yaml',  type=str_to_bool,
                            nargs='?', const=True, default=False)
    args = parser.parse_args()
    main(args)

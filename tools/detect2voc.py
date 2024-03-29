# -*- encoding: utf-8 -*-
'''
/* ***************************************************************************************************
*   NOTICE
*   This software is the property of Glint Co.,Ltd.. Any information contained in this
*   doc should not be reproduced, or used, or disclosed without the written authorization from
*   Glint Co.,Ltd..
***************************************************************************************************
*   File Name       : detect2voc.py
***************************************************************************************************
*    Module Name        : 
*    Prefix            : 
*    ECU Dependence    : None
*    MCU Dependence    : None
*    Mod Dependence    : None
***************************************************************************************************
*    Description        : 
*
***************************************************************************************************
*    Limitations        :
*
***************************************************************************************************
*
***************************************************************************************************
*    Revision History:
*
*    Version        Date            Initials        CR#                Descriptions
*    ---------    ----------        ------------    ----------        ---------------
*     1.0.0       2023-12-08            Neo                         
****************************************************************************************************/
'''
import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from loguru import logger
from yolox.exp import get_exp
from xml.dom.minidom import Document
from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, get_model_info, postprocess, vis
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# G_CLASS_NAME = ['zw', 'ss', 'yy2-yz', 'qikong_qw', 'qq', 'hh', 'yy2-qw', 'qikong', 'sz', 'NG', 'csb']
G_CLASS_NAME = (# 9
'dl',
'hh',
'mh',
'qikong',
'qq',
'ss',
'sz',
'yy',
'zw',
)

DATA_TYPE = 'train'
save_ann_dir = f'/home/glint/xzwang/AnalysisData/{DATA_TYPE}/ann'
save_img_dir_cof = f'/home/glint/xzwang/AnalysisData/{DATA_TYPE}/images_conf'
save_img_dir = f'/home/glint/xzwang/AnalysisData/{DATA_TYPE}/images'

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("-demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("--path", default=f"/home/glint/xzwang/data/0327/images", help="path to images or video")
    # parser.add_argument("--path", default=f"/home/glint/xzwang/data/0327/{DATA_TYPE}.txt", help="path to images or video")

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        default=False,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='/home/glint/xzwang/code/YOLOX/exps/example/yolox_voc/yolox_voc_s.py',
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default='/home/glint/xzwang/code/YOLOX/YOLOX_outputs/yolox_voc_s/best_ckpt.pth', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.5, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=960, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=G_CLASS_NAME,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["ori_img"] = img.copy()

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        # detection results
        if output is None:
            lines = []
            return img, lines
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        
        vis_res, lines = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, lines

def image_demo(predictor, vis_folder, path, current_time, save_result):
    os.makedirs(save_ann_dir, exist_ok=True)
    os.makedirs(save_img_dir_cof, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = Path(path).read_text().rsplit()
    files.sort()
    index = 0
    for image_name in files:
        basename = os.path.basename(image_name)
        name = basename.split('.')[0]
        outputs, img_info = predictor.inference(image_name)
        result_image, lines = predictor.visual(outputs[0], img_info, predictor.confthre)

        if len(lines) > 0:
            generate_xml(name, lines, result_image.shape, save_ann_dir)
        oriImage = img_info.get("ori_img")

        cv2.imwrite(os.path.join(save_img_dir_cof, str(name) + '.jpg'), result_image)
        cv2.imwrite(os.path.join(save_img_dir, str(name) + '.jpg'), oriImage)
        index += 1
        if index % 10 == 0:
            # print('process index ... ', index)
            logger.info(f'process index ... {index}')
        # cv2.imshow('test', result_image)
        # ch = cv2.waitKey(0)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break

def generate_xml(name, split_lines, img_size, ann_dir):
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('AD')
    title.appendChild(title_text)
    annotation.appendChild(title)

    img_name = name + '.jpg'

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The AD Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('Info')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for split_line in split_lines:
        line = split_line.strip().split()
        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        title_text = doc.createTextNode(line[0])
        title.appendChild(title_text)
        object.appendChild(title)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(float(line[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(line[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(line[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(line[4]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
    with open(os.path.join(ann_dir, name+'.xml'), 'w') as f:
        f.write(doc.toprettyxml(indent=''))

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (1280, 960)
        # exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)


    predictor = Predictor(
        model, exp, G_CLASS_NAME, None,
        args.device, args.fp16, args.legacy,
    )

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)

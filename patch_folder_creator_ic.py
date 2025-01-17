import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image, ImageDraw
import json
from tqdm import tqdm
import argparse
import numpy as np
#note max patches:
#inria: 98px
#VOC Pascal: 118px
#python3 defense_tester_new.py --patchfile gen_patches/1p/patch_adv_list_300_613.png
#python3 defense_tester_new.py --patchfile gen_patches/2p/patch_adv_list_300_613.png --n_patches 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense_meth",default='themis',type=str,help="detection mechanism")
    parser.add_argument("--savedir",default=None,type=str,help="relative directory to save results")
    parser.add_argument("--cfg",default="cfg/yolo.cfg",type=str,help="relative directory to cfg file")
    parser.add_argument("--patchfile",default="patches/object_score.png",type=str,help="relative directory to patch file")
    parser.add_argument("--weightfile",default="weights/yolo.weights",type=str,help="path to checkpoints")
    parser.add_argument('--imgdir', default="inria/Train/pos", type=str,help="path to data")
    parser.add_argument('--dataset', default='inria', choices=('imagenet','cifar', 'unknown'),type=str,help="dataset")
    parser.add_argument("--model",default='resnet50',type=str,help="model name")
    parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
    parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. set to none for local feature")
    parser.add_argument("--patch_size", default=300, type=int, help="size of the adversarial patch")
    parser.add_argument("--max_lab",default=1,type=int,help="maximum objects allowed to be patched per image")
    parser.add_argument("--fix",action='store_true',help="hold patch size")
    parser.add_argument("--n_patches",default=1,type=int,help="number of patches")
    parser.add_argument("--p_split",default='keep',type=str,help="how patchsize is distributed to n>1 patches (keep or red)")
    parser.add_argument("--skip",default=1,type=int,help="number of example to skip")
    parser.add_argument("--filter",action='store_true',help="skip images with no detcted objects")
    parser.add_argument("--sizer",action='store_true',help="only estimate max patch size")
    parser.add_argument("--save",action='store_true',help="save results to txt")
    parser.add_argument("--cropper",action='store_true',help="only crop images (no patch applying)")
    parser.add_argument("--lim",default=1000000,type=int,help="limit on number of images/frames to process")
    parser.add_argument("--p_name", default='proper_patched/', type=str, help='name of patched folder')
    args = parser.parse_args()

    print("Setting everything up")
    if args.dataset == 'imagenet':
            ds_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])
    elif args.dataset == 'cifar':
        ds_transforms = transforms.Compose([
        transforms.Resize(192, interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor()])

    imgdir = args.imgdir


    #labdir = "inria/Train/"
    cfgfile = args.cfg
    weightfile = args.weightfile
    patchfile = args.patchfile
    #patchfile = "/home/wvr/Pictures/individualImage_upper_body.png"
    #patchfile = "/home/wvr/Pictures/class_only.png"
    #patchfile = "/home/wvr/Pictures/class_transfer.png"
    #max_patches=args.n_patches
    if args.savedir is None:
        savedir = args.imgdir#args.savedir + args.dataset
    else:
        savedir = args.savedir
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    args.patch_size=int(args.patch_size/np.sqrt(args.n_patches))#+1)
    patch_size = args.patch_size
    #"""
    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    #"""
    #adv_patch_cpu=torch.zeros((3,patch_size,patch_size))

    adv_patch = adv_patch_cpu.cuda()

    clean_results = []
    noise_results = []
    patch_results = []
    ious={'clean':[], 'random':[], 'adversarial':[]}

    #THEMIS
    clean_corr=0
    detected=0
    success_atk=0
    iou_thresh=0.5



    kount=0
    lim=args.lim
    #print(os.listdir(imgdir))
    #input('well?')
    objs_in_frame=[]
    if args.sizer:
        sizes=[]
    for imgfile in tqdm(os.listdir(imgdir)[:min(args.lim, len(os.listdir(imgdir)))]):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png') or  imgfile.endswith('.JPEG'):
            name = os.path.splitext(imgfile)[0]    #image name w/o extension
            txtname = name + '.txt'
            #if '002582' not in name:
            #    continue
            #txtpath = os.path.abspath(os.path.join(labdir, 'yolo-labels/', txtname))
            # open beeld en pas aan naar yolo input size
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')

            #transform = transforms.ToTensor()
            padded_img = ds_transforms(img)
            w,h = padded_img.shape[1:]

            px, py = np.random.randint(0,w-args.patch_size), np.random.randint(0,h-args.patch_size)
            px2, py2 = w-args.patch_size-1-px, h-args.patch_size-1-py
            #input([px,py,px2,py2])
            if args.n_patches>=1:
                padded_img[:,px:px+args.patch_size, py:py+args.patch_size] = adv_patch_cpu
            if args.n_patches>=2:
                padded_img[:,px2:px2+args.patch_size, py2:py2+args.patch_size] = adv_patch_cpu
            if args.n_patches==4:
                padded_img[:,px2:px2+args.patch_size, py:py+args.patch_size] = adv_patch_cpu
                padded_img[:,px:px+args.patch_size, py2:py2+args.patch_size] = adv_patch_cpu

            p_img_pil = transforms.ToPILImage('RGB')(padded_img)
            properpatchedname = name + ".png"
            deer=os.path.join(savedir, args.p_name)
            if not os.path.exists(deer):
                os.makedirs(deer)
            p_img_pil.save(deer +  properpatchedname)

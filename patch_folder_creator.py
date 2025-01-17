import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils.utils import *
from helper import *
from darknet import *
from load_data import PatchTransformer, PatchApplier, InriaDataset
import json
from tqdm import tqdm
import argparse
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
    parser.add_argument('--dataset', default='unknown', choices=('inria','COCO', 'unknown'),type=str,help="dataset")
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

    #deprecated, keep as unknown always
    print("Setting everything up")
    if args.dataset == 'inria':
        imgdir = "inria/Train/pos"
    elif args.dataset == 'COCO':
        imgdir = "COCO/val"
    elif args.dataset == 'unknown':
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

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer(n_patches= args.n_patches, size_fix=args.fix).cuda()

    batch_size = 1
    max_lab = 14
    img_size = darknet_model.height
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
    beta, theta = 0.75, 0.85 #paper vals: 0.75, 0.85
    wsizepx=int(args.patch_size)#+1)
    wsize=int(wsizepx/4)#Themis: this goes in the feature map
    #print("Done")
    #Loop over cleane beelden
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
            w,h = img.size
            #print(w)
            #print(h)
            #input("yes")
            if w==h:
                padded_img = img
            else:
                dim_to_pad = 1 if w<h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                    padded_img.paste(img, (0, int(padding)))
            resize = transforms.Resize((img_size,img_size))
            padded_img = resize(padded_img)
            if args.cropper:
                cleanname = name + ".png"
                deer=os.path.join(savedir, 'clean/')
                if not os.path.exists(deer):
                    os.makedirs(deer)
                padded_img.save(deer+cleanname)
                continue


            boxes, feature_map = do_detect(darknet_model, padded_img, 0.4, 0.4, True)
            """
            for b in range(batch_size):
                fm=torch.sum(feature_map[0],dim=0)
                #print(boxes.shape)
                print(fm.shape)
                input('hoold it')
            """
            #boxes = nms(boxes, 0.4)
            clean_boxes=[]
            for box in boxes:
                if True:#box[6]==0:
                    clean_boxes.append(box)
            #clean_boxes=boxes

            objs_in_frame.append(len(clean_boxes))
            if not len(clean_boxes) and args.filter:
                continue
            cleanname = name + ".png"
            deer=os.path.join(savedir, 'clean/')
            if not os.path.exists(deer):
                os.makedirs(deer)
            padded_img.save(deer+cleanname)

            clean_boxes=clean_boxes[:min(args.max_lab, len(clean_boxes))]
            cbb=[]
            for cb in clean_boxes:
                cbb.append([T.detach() for T in cb])
            """
            if len(clean_boxes)<max_patches:
                print("lotto no!")
                print(len(clean_boxes))
                print(imgfile)
                kount=kount-1
                continue
            """
            deer=os.path.join(savedir, 'clean/', 'yolo-labels/')
            if not os.path.exists(deer):
                os.makedirs(deer)
            txtpath = deer + txtname
            textfile = open(txtpath,'w+')
            for i, box in enumerate(clean_boxes):
                #if i>=max_patches:
                #    break
                cls_id = box[6]
                if True:#(cls_id == 0):   #if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    clean_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                     y_center.item() - height.item() / 2,
                                                                     width.item(),
                                                                     height.item()],
                                          'score': box[4].item(),
                                          'category_id': 1})
            textfile.close()
            #lees deze labelfile terug in als tensor
            if os.path.getsize(txtpath):       #check to see if label file contains data.
                label = np.loadtxt(txtpath)
            else:
                #warning: means there are not objects detected, so it is pointless to attack
                print('I DONT WANNA LIVE LIKE THIS')
                print(imgfile)
                continue
                #label = np.ones([5])
                label=np.array([-1,1,1,1,1])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)


            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).cuda()

            a = lab_fake_batch[:, :, 3] * img_size
            b = lab_fake_batch[:, :, 4] * img_size
            target_size = max(torch.sqrt((a ** 2) + (b ** 2)).mul(1/5).flatten().detach().cpu().numpy())
            if args.sizer:
                sizes.append(target_size)
                continue
            #print(target_size)
            #input('behiold')
            #wsize=int(target_size)
            #transformeer patch en voeg hem toe aan beeld
            adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=False, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + ".png"
            deer=os.path.join(savedir, args.p_name)
            if not os.path.exists(deer):
                os.makedirs(deer)
            p_img_pil.save(deer +  properpatchedname)

            #genereer een label file voor het beeld met sticker
            txtname = properpatchedname.replace('.png', '.txt')
            deer=os.path.join(savedir, args.p_name, 'yolo-labels/')
            if not os.path.exists(deer):
                os.makedirs(deer)
            txtpath = deer + txtname
######################################################################################################################################
            import warnings
            import copy
            #list_str=adv_dataset.split('/')[0]+'/'+adv_dataset.split('/')[1].replace('adv','loc')
            #attacked_locs = joblib.load('./dump/' + list_str)
            warnings.filterwarnings("ignore")
            #model = model.to(device)
            #model.eval()
            all_clean, all_det, all_atk = [], [], []
            boxes, feature_map  = do_detect(darknet_model, p_img_pil, 0.4, 0.4, True)
            adv_boxes=boxes
            adb = []
            for ab in adv_boxes:
                adb.append([T.detach() for T in ab])
#Sanity check: check undefended performance....
######################################################################################################################################
            for i, box in enumerate(clean_boxes):

                ious['clean'].append(best_iou(cbb, [T.detach() for T in clean_boxes[i]]))
                #print("iou rand:")
                #print("iou adv:")

                iou_adv=best_iou(adb, [T.detach() for T in clean_boxes[i]])
                ious['adversarial'].append(iou_adv)
                if iou_adv>=iou_thresh:#clean_pred==labels[i]:
                    clean_corr = clean_corr + 1
                else:
                    success_atk = success_atk + 1
            #input('take a breather my boy')
            torch.cuda.empty_cache()
    if args.sizer:
        print("largest patch in dataset: ")
        print(np.max(sizes))
        print("out ouf {} patches".format(len(sizes)))
        print("with mean size: ")
        print(np.mean(sizes))
    else:
        print("stats: ")

        print("clean avg iou:")
        print(np.mean(ious['clean']))
        print("adv patch avg iou:")
        print(np.mean(ious['adversarial']))
        if len(ious['clean']):
            print("adv patch misdetection rate:")
            print(sum([x<iou_thresh for x in ious['adversarial']])/len(ious['clean']))

            kount = len(ious['clean'])
            #print("Parameters:\nBeta: "+str(beta)+" Theta: "+str(theta)+" Wsize: " + str(wsize))
            line1="Unsuccesful Attacks:" + str(clean_corr/kount)#len(os.listdir(imgdir)))
            line2="Detected Attacks:" + str(detected/kount)#len(os.listdir(imgdir)))
            line3="Successful Attacks:"+str(success_atk/kount)#len(os.listdir(imgdir)))
            print(line1)
            print(line2)
            print(line3)
            print("------------------------------")
            print("lengths!")
            print(len([x!=0 for x in objs_in_frame]))
            print(np.mean(objs_in_frame))
        else:
            print('what a life')

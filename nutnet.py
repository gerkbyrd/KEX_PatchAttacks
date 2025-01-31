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
from helper import *
from darknet import *
import patchFilter as pf

#from defense import YOLO_wrapper,ObjSeekerModel
import json
from tqdm import tqdm
import argparse
import copy
from timeit import default_timer as timer
#from mean_average_precision import MetricBuilder
#metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=80)
#python3 defense_tester_new.py --patchfile gen_patches/1p/patch_adv_list_300_613.png
#python3 defense_tester_new.py --patchfile gen_patches/2p/patch_adv_list_300_613.png --n_patches 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",default='clustering',type=str,help="detection mechanism (clustering or ez)")
    parser.add_argument("--savedir",default="nutnet_results/",type=str,help="relative directory to save results")
    parser.add_argument("--cfg",default="cfg/yolo.cfg",type=str,help="relative directory to cfg file")
    parser.add_argument("--patchfile",default="patches/object_score.png",type=str,help="relative directory to patch file")
    parser.add_argument("--weightfile",default="weights/yolo.weights",type=str,help="path to checkpoints")
    parser.add_argument('--imgdir', default="inria/Train/pos", type=str,help="path to data")
    parser.add_argument('--patch_imgdir', default="inria/Train/pos", type=str,help="path to patched data")
    parser.add_argument('--nopedir', default=None, type=str,help="path to data to avoid")
    parser.add_argument('--yepdir', default=None, type=str,help="path to data to use")
    parser.add_argument("--save_scores",action='store_true',help="save detection scores")
    parser.add_argument("--save_outcomes",action='store_true',help="save recovery outcomes")
    parser.add_argument("--localization",action='store_true',help="loc. results")
    parser.add_argument("--effective_files",default=None,type=str,help="file with list of effective adv examples")
    parser.add_argument("--uneffective",action='store_true',help="only uneffective attacks")
    parser.add_argument("--eval_class",action='store_true',help="match class for iou evaluation")

    parser.add_argument("--n_patches",default=1,type=int,help="number of patches (just an informative string)")
    parser.add_argument('--dataset', default='inria',type=str,help="dataset")
    parser.add_argument("--model",default='resnet50',type=str,help="model name")
    parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
    parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. set to none for local feature")
    parser.add_argument("--max_lab",default=1,type=int,help="maximum objects allowed to be patched per image")

    parser.add_argument("--skip",default=1,type=int,help="number of examples  to skip")
    parser.add_argument("--save",action='store_true',help="save results to txt")
    parser.add_argument("--lim",default=1000000,type=int,help="limit on number of images/frames to process")


    parser.add_argument("--clean",action='store_true',help="use clean images")
    parser.add_argument("--bypass",action='store_true',help="skip detection")
    parser.add_argument("--visualize",action='store_true',help="save obj. detection output images")
    parser.add_argument('--num-line', type=int, default=30, help='number of lines $k$')

    # objectseeker argumenets
    # we call vanilla object detector "base detector"
    parser.add_argument("--box_num", default=8, type=int,help="NutNet boxnum")
    parser.add_argument('--thresh1', type=float, default=0.125, help='coarse NutNet threshold')
    parser.add_argument('--thresh2', type=float, default=0.2, help='fine NutNet threshold')


    parser.add_argument('--device', type=str, default='cuda', help='cuda or CPU')
    parser.add_argument("--performance",action='store_true',help="save recovery performance (time) per frame")


    args = parser.parse_args()

    print("Setting everything up")
    imgdir=args.imgdir
    patchdir=args.patch_imgdir

    #labdir = "inria/Train/"
    cfgfile = args.cfg
    weightfile = args.weightfile
    patchfile = args.patchfile
    #patchfile = "/home/wvr/Pictures/individualImage_upper_body.png"
    #patchfile = "/home/wvr/Pictures/class_only.png"
    #patchfile = "/home/wvr/Pictures/class_transfer.png"
    #max_patches=args.n_patches
    #mn, std = args.scale_mean, args.scale_var

    device='cuda'
    #"""
    savedir = args.savedir# + args.dataset
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    #"""
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    nutnet=pf.nutnet_getter(inp_dim=416, device='cuda', box_num=args.box_num, thresh1=args.thresh1, thresh2=args.thresh2)

    #model.load_weights(weightfile)
    #model=model.eval().cuda()

    batch_size = 1
    max_lab=14
    img_size = darknet_model.height


    ious={'clean':[], 'random':[], 'adversarial':[]}

    #THEMIS
    clean_corr=0
    detected=0
    success_atk=0
    iou_thresh=0.5

    kount=0
    lim=args.lim

    import warnings
    import copy
    warnings.filterwarnings("ignore")
    if args.save_scores:
        score_array=[]
    if args.save_outcomes:
        outcome_array=[]
    if args.effective_files!=None:
        eff_files=list(np.load(os.path.join(patchdir, args.effective_files)))
        eff_files=[x.split('.')[0] for x in eff_files]
    else:
        eff_files = None
    if args.performance:
        perf_array=[]
    all_atk, mask_ious=[],[]

    deer=os.path.join(args.savedir)
    if not os.path.exists(deer):
        os.makedirs(deer)

    valds=sorted(os.listdir(imgdir)[:min(args.lim, len(os.listdir(imgdir)))])
    for imgfile in tqdm(valds):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            #print(imgfile)
            nameee=imgfile.split('.')[0]
            if 'predictions' in nameee:
                continue
            if (eff_files != None and nameee not in eff_files and not args.uneffective) or (eff_files != None and args.uneffective and nameee in eff_files):
                continue
            patchfile = os.path.abspath(os.path.join(patchdir, imgfile))
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))

            padded_img = Image.open(imgfile)
            feature_size_x, feature_size_y=padded_img.size

            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)

            clean_boxes, feature_map = do_detect(darknet_model, img_fake_batch, 0.4, 0.4, True, direct_cuda_img=True)
            #objs_in_frame.append(len(clean_boxes))
            #clean_boxes=clean_boxes[:min(args.max_lab, len(clean_boxes))]
            if not(len(clean_boxes)):
                continue
            kount=kount+1
            cbb=[]
            for cb in clean_boxes:
                cbb.append([T.detach() for T in cb])
            if not args.clean:
                patched_img = Image.open(patchfile)
                patched_img = transform(patched_img).cuda()
                p_img = patched_img.unsqueeze(0)
                og_outs = []
                candigatos=[]
                adv_boxes, feature_map  = do_detect(darknet_model, p_img, 0.4, 0.4, True, direct_cuda_img=True)
                adb = []
                for ab in adv_boxes:
                    adb.append([T.detach() for T in ab])
            else:
                p_img=img_fake_batch#dont matter cos we use feature map only
                candigatos=[]
                adb=cbb

            if args.performance:
                start=timer()
            #print(p_img.shape)
            if not args.bypass:
                p_img_nutnet=nutnet(p_img)
            else:
                p_img_nutnet=p_img
            #input(p_img_nutnet.shape)
            nn_boxes, _ = do_detect(darknet_model, p_img_nutnet, 0.4, 0.4, True, direct_cuda_img=True)
            #input(os_boxes)
            if args.performance:
                end=timer()
                perf_array.append(end-start)
            nnbb=[]
            for nsb in nn_boxes:
                nnbb.append([T.detach().cpu() for T in nsb])

            if args.visualize:
                img=transforms.ToPILImage()(p_img_nutnet.squeeze(0))
                #img = Image.open(patchfile).convert('RGB')
                class_names = load_class_names('data/coco.names')
                plot_boxes(img, nnbb, args.savedir +'/' +nameee+'.jpg', class_names, doconv=True)

            suc_atk=False
            for i in range(len(clean_boxes)):
                ious['clean'].append(best_iou(cbb, [T.detach() for T in clean_boxes[i]]))
                best=best_iou(nnbb, [T.detach() for T in clean_boxes[i]], match_class=args.eval_class)
                ious['adversarial'].append(best)
                if best<iou_thresh:#clean_pred==labels[i]:
                    suc_atk=True
                    success_atk = success_atk + 1
                    break

            if not suc_atk:
                clean_corr+=1

            cbb=[]
            for cb in clean_boxes:
                cbb.append([T.detach() for T in cb])
            clean_map=np.array(cbb)
            clean_map[:,4]=clean_map[:,-1]
            clean_map[:,5:]=0
            if not len(nnbb):
                rec_map=np.zeros((1,6))
            else:
                rec_map=np.array(nnbb)
            class_guess=np.copy(rec_map[:,5])
            rec_map[:,5]=rec_map[:,4]
            rec_map[:,4]=class_guess
            if args.save_outcomes:
                outcome_array.append([clean_map, rec_map])
            #if not suc_atk:
            #    clean_corr=clean_corr+1
            torch.cuda.empty_cache()

    #kount = min(args.lim, len(os.listdir(imgdir)))#len(ious['clean'])
    #print("Parameters:\nBeta: "+str(beta)+" Theta: "+str(theta)+" Wsize: " + str(wsize))
    line1="Unsuccesful/Recovered Attacks:" + str(clean_corr/kount)#len(os.listdir(imgdir)))
    #line2="Detected Attacks:" + str(detected/kount)#len(os.listdir(imgdir)))
    line3="Successful Attacks:"+str(success_atk/kount)#len(os.listdir(imgdir)))
    print(line1)
    #print(line2)
    print(line3)
    #print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")
    print("------------------------------")
    #print(lines)
    #print(np.mean(all_atk))
    #print(np.mean(mask_ious))

    if args.save:
        deer=os.path.join(args.savedir)
        if not os.path.exists(deer):
            os.makedirs(deer)
        txtpath = args.savedir + '_npatches_' + str(args.n_patches) + '_boxnum_'+ str(args.box_num) + '_thr1_' + str(args.thresh1) + '_thr2_' + str(args.thresh2)
        if args.clean:
            txtpath = txtpath + '_clean'
        with open(txtpath + '.txt', 'w+') as f:
            f.write('\n'.join([line1, line2, line3, "------------------------------"]))

    if args.performance:
        deer=os.path.join(args.savedir)
        if not os.path.exists(deer):
            os.makedirs(deer)
        fname=deer + '_npatches_' + str(args.n_patches) + '_boxnum_'+ str(args.box_num) + '_thr1_' + str(args.thresh1) + '_thr2_' + str(args.thresh2) + '_perfs'
        if args.clean:
            fname = fname + '_clean'
        with open(fname + '.npy', 'wb') as f:
            np.save(f, np.array(perf_array))

    if args.save_outcomes:
        deer=os.path.join(args.savedir)
        if not os.path.exists(deer):
            os.makedirs(deer)
        fname=deer + '_npatches_' + str(args.n_patches) + '_boxnum_'+ str(args.box_num) + '_thr1_' + str(args.thresh1) + '_thr2_' + str(args.thresh2) + '_scores'
        if args.clean:
            fname = fname + '_clean'
        with open(fname + '.npy', 'wb') as f:
            np.save(f, np.array(outcome_array))

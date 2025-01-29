"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm
import argparse

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
#from tensorboardX import SummaryWriter
import subprocess

#import patch_config
import os
import sys
import time

def generate_patch(type, n_patches=1, patch_size=300):
    """
    Generate a random patch as a starting point for optimization.

    :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
    :return:
    """
    if type == 'gray':
        adv_patch_cpu = torch.full((3, patch_size, patch_size), 0.5)
    elif type == 'random':
        adv_patch_cpu = torch.rand((3, patch_size, patch_size))

    return adv_patch_cpu

def train(args):
    #config = patch_config.patch_configs[mode]()
    darknet_model = Darknet(args.cfgfile)
    darknet_model.load_weights(args.weightfile)
    darknet_model = darknet_model.eval().cuda() # TODO: Why eval?
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer(n_patches=args.n_patches).cuda()
    if args.loss_target=='obj':
        loss_target=lambda obj, cls: obj
    elif args.loss_target=='cls':
        loss_target=lambda obj, cls: cls
    else:
        loss_target=lambda obj, cls: obj * cls
    prob_extractor = MaxProbExtractor(0, 80, loss_target=loss_target).cuda()
    nps_calculator = NPSCalculator(args.printfile, args.patch_size).cuda()
    total_variation = TotalVariation().cuda()
    n_patches=args.n_patches

    #DATA_DIR=os.path.join('carla_gen_patches/' + str(n_patches) + 'p/')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    img_size = darknet_model.height
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    max_lab = 14

    time_str = time.strftime("%Y%m%d-%H%M%S")

    # Generate stating point
    #Option 1: gray patch
    if args.patch_checkpoint is None:
        adv_patch_cpu = generate_patch("gray")
    #Option 2: some existing image (possibly an adv. patch to be trained some more)
    else:
        patch_img_m = PIL.Image.open(args.patch_checkpoint).convert('RGB')
        tf = transforms.Resize((args.patch_size, args.patch_size))
        patch_img_m = tf(patch_img_m)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img_m)

    adv_patch_cpu.requires_grad_(True)

    train_loader = torch.utils.data.DataLoader(InriaDataset(args.imgdir, args.labdir, args.max_lab, img_size,
                                                            shuffle=False),#True),
                                               batch_size=batch_size,
                                               shuffle=False,#True,
                                               num_workers=1)
    epoch_length = len(train_loader)
    print(f'One epoch is {len(train_loader)}')

    optimizer = optim.Adam([adv_patch_cpu], lr=args.start_learning_rate, amsgrad=True)
    scheduler_factory=lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
    scheduler = scheduler_factory(optimizer)

    #initialize the loss and early stopping counter
    best=np.inf
    patience=0

    et0 = time.time()
    for epoch in range(n_epochs):
        ep_det_loss = 0
        ep_nps_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        for i_batch, (img_batch, lab_batch, pth_i) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            #if i_batch>0:
            #    break

            img_batch = img_batch.cuda()
            lab_batch = lab_batch.cuda()
            #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
            adv_patch = adv_patch_cpu.cuda()
            adv_batch_t = patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True)
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))

            #img = p_img_batch[0, :, :,]
            #img = transforms.ToPILImage()(img.detach().cpu())
            #img.show()


            output, _ = darknet_model(p_img_batch)
            max_prob = prob_extractor(output)
            nps = nps_calculator(adv_patch)
            tv = total_variation(adv_patch)


            nps_loss = nps*0.01
            tv_loss = tv*2.5
            det_loss = torch.mean(max_prob)
            loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

            ep_det_loss += det_loss.detach().cpu().numpy()
            ep_nps_loss += nps_loss.detach().cpu().numpy()
            ep_tv_loss += tv_loss.detach().cpu().numpy()
            ep_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

            #bt1 = time.time()
            #bt0 = time.time()
        et1 = time.time()
        ep_det_loss = ep_det_loss/len(train_loader)
        ep_nps_loss = ep_nps_loss/len(train_loader)
        ep_tv_loss = ep_tv_loss/len(train_loader)
        ep_loss = ep_loss/len(train_loader)

        #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        #plt.imshow(im)
        #plt.savefig(f'pics/{time_str}_{config.patch_name}_{epoch}.png')

        scheduler.step(ep_loss)
        if (epoch+1)%25==0 or epoch==0 or True:
            print('  EPOCH NR: ', epoch+1, ' EPOCH LOSS: ', ep_loss, ' IMG NO: ', i_batch)
            #writer.add_image('patch_adv_{}_{}'.format(config.patch_size, i_batch), adv_patch_cpu, epoch)
        #print('  DET LOSS: ', ep_det_loss)
        #print('  NPS LOSS: ', ep_nps_loss)
        #print('   TV LOSS: ', ep_tv_loss)
            print('EPOCH TIME: ', et1-et0)
        if ep_loss<best:
            patience=0
            best=ep_loss
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            im.save(args.savedir + '/universal_patch'.format(args.patch_size) +'.png')
        else:
            patience=patience+1
        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
        torch.cuda.empty_cache()
        et0 = time.time()
        if patience>=args.patience:
            print('Early stopping at epoch ', epoch+1)
            break
            #joblib.dump(adv_patch.cpu().detach().numpy(), os.path.join(DATA_DIR,'patch_adv_list_{}_{}p_'.format(args.patch_size, args.n_patches) + args.p_split + '.z'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default='paper_obj',type=str,help="detection mechanism")
    parser.add_argument("--savedir",default="testing/",type=str,help="relative directory to save results")
    parser.add_argument("--cfg",default="cfg/yolo.cfg",type=str,help="relative directory to cfg file")
    #parser.add_argument("--patchfile",default="patches/object_score.png",type=str,help="relative directory to patch file")
    parser.add_argument("--weightfile",default="weights/yolo.weights",type=str,help="path to checkpoints")
    parser.add_argument("--cfgfile",default="cfg/yolo.cfg",type=str,help="path to checkpoints")
    parser.add_argument('--imgdir', default="inria/Train/pos/clean", type=str,help="path to data")
    parser.add_argument('--labdir', default="inria/Train/pos/clean/yolo-labels", type=str,help="path to labels")
    parser.add_argument('--dataset', default='inria', choices=('inria','COCO'),type=str,help="dataset")

    parser.add_argument("--model",default='resnet50',type=str,help="model name")
    parser.add_argument("--patch_checkpoint",default=None,type=str,help="file to start training from")
    parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
    parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. set to none for local feature")
    parser.add_argument("--loss_target",default='obj',type=str,help="patch attack objective")
    parser.add_argument("--batch_size", default=16, type=int, help="betch size")
    parser.add_argument("--patch_size", default=300, type=int, help="size of the adversarial patch")
    parser.add_argument("--max_lab", default=14, type=int, help="size of the adversarial patch")
    parser.add_argument("--patience", default=10, type=int, help="epochs with no improvement for early stopping")
    parser.add_argument("--n_epochs", default=20000, type=int, help="size of the adversarial patch")

    parser.add_argument("--fix",action='store_true',help="hold patch size")
    parser.add_argument("--n_patches",default=1,type=int,help="number of patches")
    parser.add_argument("--start_learning_rate",default=0.03,type=float,help="initial learning rate for optimizer")
    parser.add_argument('--printfile', default="non_printability/30values.txt", type=str,help="path to printability file")


    parser.add_argument("--p_split",default='keep',type=str,help="how patchsize is distributed to n>1 patches (keep or red)")
    parser.add_argument("--skip",default=1,type=int,help="number of example to skip")
    parser.add_argument("--save",action='store_true',help="save results to txt")
    args = parser.parse_args()
    train(args=args)
    #trainer = PatchTrainer(args=args, n_patches=args.n_patches)
    #trainer.train()

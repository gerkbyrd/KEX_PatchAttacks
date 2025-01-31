## KEX Project: Patch Attacks
directories:
- cfg: contains cfg files for possible object detection victim models (YOLOv2 by default)
- checkpoints: general folder for other model weights (for now it contains the weights for ResNet50 trained for CIFAR-10)
- data: contains a small amount of examples for public benchmark datasets for object detection (INRIA and Pascal VOC) and image classification (Imagenet and CIFAR-10)
- nets: torch model for the image classification victim model (ResNet50)
- patches: png/jpg images of trained patches ready to be applied
- utils: contains utils.py, a file with helper functions of object detection
- weights: the weights for YOLOv2 should be placed in this folder (IMPORTANT! this weight file is not in the repository due to file size)

files:
- cfg.py, darknet.py, region_loss.py: files required to run YOLOv2 using PyTorch.
- helper.py: various helper functions to perform object detection.
- patch_folder_creator.py, patch_folder_creator_ic.py: code to apply already trained patches (saved in the "patches" folder) on clean images pertaining to object detection and image classification.
- load_data.py, median_pool.py: files required to run the code to train/apply patch attacks.
- PatchAttacker.py: basic file to run non-universal attacks on image classification (may be irrelevant to the project...)

This code is based on the following publicly available repositories:

- https://github.com/Zhang-Jack/adversarial_yolo2
- https://github.com/inspire-group/PatchGuard/tree/master

For attacks on CIFAR-10 it is necessary to download the resnet50_192_cifar.pth file from https://github.com/inspire-group/PatchGuard/tree/master and place it in the checkpoints folder (already done!)

For attacks on INRIA and Pascal VOC it is necessary to follow the instructions on https://github.com/Zhang-Jack/adversarial_yolo2 to download the yolo.weights file into the weights folder (IMPORTANT TO DO)


## EXAMPLE COMMANDS - APPLY PATCHES

Here are some examples on how to apply already trained patches from available in "patches/". For the following commands, patched version of the clean images located at the folder specified using "--imgdir FOLDER" will be saved at "FOLDER/NAME" where the NAME is provided using "--p_name NAME".

Apply the "Princess (Frozen)" patch on the INRIA images:
```
python patch_folder_creator.py --imgdir data/inria/clean/ --n_patches 1 --patchfile patches/newpatch.PNG --p_name test/
```
For double patches:
```
python patch_folder_creator.py --imgdir data/inria/clean/ --n_patches 2 --patchfile patches/newpatch.PNG --p_name test/
```
Note that the last commands will apply patches on at most one object per image, one can change the max_lab argument (by default 1) to change this. For double patches applied on up to 3 objects per image (see --max_lab argument!):
```
python patch_folder_creator.py --imgdir data/inria/clean/ --n_patches 1 --patchfile patches/newpatch.PNG --p_name test/ --max_lab 3
```

For image classification, we have to also specify what dataset we are using and the size of the patch. We do not need to define the size for object detection because it is resized depending on the attacked object, but in image classification the patch is randomly placed somewhere in the image. Note also that the size of the patch when we input "--patch_size SIZE" is SIZE x SIZE. Hence, for 32x32 single patches on imagenet using the "Electric Guitar" patch:
```
patch_folder_creator_ic.py --imgdir data/imagenet/clean/ --dataset imagenet --n_patches 1 --patchfile patches/imgnet_patch.PNG --p_name test/ --patch_size 32
```
For double patches:
```
patch_folder_creator_ic.py --imgdir data/imagenet/clean/ --dataset imagenet --n_patches 2 --patchfile patches/imgnet_patch.PNG --p_name test/ --patch_size 32
```
And quadruple patches:
```
patch_folder_creator_ic.py --imgdir data/imagenet/clean/ --dataset imagenet --n_patches 4 --patchfile patches/imgnet_patch.PNG --p_name test/ --patch_size 32
```

Note that in all examples, the area of the single patch attack is maintained when using multiple patches, that is, multiple patches are smaller to ensure that the attacked area is the same.

## EXAMPLE COMMANDS - TRAIN PATCHES

Here are some examples on how to train a patch from scratch. For the following examples commands, the clean images the train patch will attack are located at the folder specified using "--imgdir FOLDER", the labels indicating objects detected in the clean images by the victim model are at the folder specified using "--labdir FOLDER". The trained patch will be saved at "FOLDER/universal_patch.png" specified provided using "--savedir FOLDER/". 

Train a patch on the clean INRIA images (patience represents the number of epochs without imporvements since the best patch so far before stopping):
```
python train_patch.py --patience 100 --imgdir data/inria/clean --labdir data/inria/yolo-labels/ --savedir testing/
```

This should yield a patch similar to the one in "testing/test_patch.png".


## EXAMPLE COMMANDS - DEFENSE, EVALUATION, AND VISUALIZATION

Here are some examples on how evaluate the effectiveness of patch attacks and defenses (this code is built around the NutNet defense). For the following example commands, the clean images are located at the folder specified using "--imgdir FOLDER", and their attacked versions (with the same names) at the folder specified using "--patch_imgdir FOLDER". Visualziations of the obejct detection results can be saved in the folder specified by using "--savedir FOLDER/" and the flag "--visualize". See the "nutnet_results" folder for some examples of visualization. The "--clean" flag indicates that the code will run only on clean images (this is equivalent to using the clean image folder in both "--imgdir" and "--patch_imgdir"), and the "--bypass" flag indicates that the NutNet defense should be bypassed, thus evaluating the undefended model. The script will output "Recovered/unsuccessful attacks" and "successful attacks": the former refer to the proportion of cases where the patch attack was not effective or properly defended, and the latter to the proportion of cases where the patches were effective and defeated NutNet (the last part is only true if the evaluation does not use the "--bypass" flag). 

IMPORTANT: THIS CODE IS FOCUSED ON OBJ. DETECTION, THUS ONLY INRIA AND VOC SHOULD BE USED.

To get results of the victim model on clean images of the INRIA dataset with visualizations (saved to undefended_clean/):
```
python nutnet.py --imgdir data/inria/clean --patch_imgdir data/inria/1p/ --bypass --visualize --clean --savedir undefended_clean/
```
This should yield results like the ones in "nutnet_results/".

To get results of the undefended victim model on images of the INRIA dataset with single-patch attacks with visualizations (saved to undefended_1p/):
```
python nutnet.py --imgdir data/inria/clean --patch_imgdir data/inria/1p/ --bypass --visualize --savedir undefended_1p/
```

To get results of the victim model defended by NutNet on images of the INRIA dataset with single-patch attacks with visualizations (saved to nutnet_1p/):
```
python nutnet.py --imgdir data/inria/clean --patch_imgdir data/inria/1p/ --visualize --savedir nutnet_1p/
```


import os

import albumentations as albu
#import cv2# not using due to issue in loading using cv2.imread for few images
import keras
from keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def read_image(img_path):
    #img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = keras.preprocessing.image.load_img(img_path)
    img = np.array(img)#, dtype="float32" this is done by normalize
    return img


def read_mask(mask_path):
    #mask_path = mask_list[idx]
    #mask = cv2.imread(mask_path)
    mask = keras.preprocessing.image.load_img(mask_path, color_mode = "grayscale")
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #mask = np.array(mask) -1# (0,1,2) instead of (1,2,3)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask -1
    return mask


def augment_image_and_mask(image, mask, aug):
    augmented = aug(image=image, mask=mask)
    image_augmented = augmented['image']
    mask_augmented = augmented['mask']
    return image_augmented, mask_augmented


def normalize_img(img):
    max_pixval = [255.0 if np.max(img)>128 else np.max(img)]
    #print("Dividing by {}".format(max_pixval[0]))
    img = img / np.max(img)# divide by maximum
    #img[img > 1.0] = 1.0
    #img[img < 0.0] = 0.0
    return img

class OxfordPetsData(keras.utils.Sequence):
    """Helper class to iterate over teh data as numpy arrays """

    def __init__(self, batch_size, img_size, input_img_paths, input_mask_paths,augmentation=None):
        self.batch_size = batch_size
        self.img_size = (img_size,img_size)
        self.input_img_paths = input_img_paths
        self.input_mask_paths = input_mask_paths
        self.aug = augmentation# user supplied augumentation function "i.e. albumentations"

    def __len__(self):
        '''data len = 7390; bs = 32; len = 7390// 32 ~= 230'''
        return len(self.input_mask_paths) // self.batch_size

    def __getitem__(self, idx):
        """Return tuple(input, target) or (img, mask) correspondidng to batch #idx
        single Call to getitem will return batch_size length of data"""
        startIndex = idx * self.batch_size
        stopIndex = startIndex + self.batch_size
        batch_ip_img_paths = self.input_img_paths[startIndex: stopIndex]
        batch_ip_mask_paths = self.input_mask_paths[startIndex: stopIndex]

        # both input_img and target_img will have size of img_size=(160,160)
        #x shape =(32,H,W,3) NHWC format  i.e. 4D tensor
        batch_imgs = np.zeros((self.batch_size,)+ self.img_size + (3,), dtype = "float32")
        # y shape =(N,H,W,c=1)
        batch_masks = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for ii in range(self.batch_size):
            img = read_image(batch_ip_img_paths[ii])
            mask = read_mask(batch_ip_mask_paths[ii])
            if self.aug!=None:
                img, mask = augment_image_and_mask(img, mask, self.aug)
            img = normalize_img(img)
            batch_imgs[ii] = img
            batch_masks[ii] = mask
        return batch_imgs,batch_masks
#end of class
def main():
    # constants
    IMG_SIZE = 256
    BATCH_SIZE = 4
    NUM_CLASSES = 3

    # read list of filenames from dir
    imgs_dir ="D:\\prjs\\oxf_unet\\data\\train\\images"
    masks_dir ="D:\\prjs\\oxf_unet\\data\\train\\masks"
    train_image_list = sorted([os.path.join(imgs_dir,fname) for fname in os.listdir(imgs_dir)])
    train_mask_list = sorted([os.path.join(masks_dir,fname) for fname in os.listdir(masks_dir)])

    # shuffle files with a fixed seed for reproducibility
    #idx = np.arange(len(image_list))
    #np.random.seed(1)
    #np.random.shuffle(idx)
    #image_list = [image_list[i] for i in idx]
    #mask_list = [mask_list[i] for i in idx]

    # define image augmentation operations for train and test set
    aug_train = albu.Compose([
        albu.Blur(blur_limit=3),
        albu.HorizontalFlip(p=0.5),#(-0.9, 1.2)
        #albu.Normalize(),
        albu.RandomBrightnessContrast(contrast_limit=0.3,brightness_limit=0.3,brightness_by_max=True),
        #albu.RandomGamma(),
        albu.augmentations.transforms.Resize(height=IMG_SIZE, width=IMG_SIZE),
        albu.RandomSizedCrop((IMG_SIZE - 50, IMG_SIZE - 1), IMG_SIZE, IMG_SIZE)
    ])

    aug_test = albu.Compose([
        albu.augmentations.transforms.Resize(height=IMG_SIZE, width=IMG_SIZE)
    ])

    #
    #(batch_size, img_size, input_img_paths, input_mask_paths, augmentation = None)
    # construct train and test data generators
    train_generator = OxfordPetsData(
        input_img_paths=train_image_list,
        input_mask_paths =train_mask_list,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augmentation=aug_train)

    test_generator = OxfordPetsData(
        input_img_paths=train_image_list,
        input_mask_paths=train_mask_list,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augmentation=aug_test)

    img_batch, mask_batch = train_generator[0]
    disp_scaled(img_batch, mask_batch)

def get_train_test_split():
    input_dir = "D:/prjs/oxf_unet/data/images/"
    target_dir = "D:/prjs/oxf_unet/data/annotations/trimaps/"

    input_img_paths = sorted([os.path.join(input_dir, fname)
                              for fname in os.listdir(input_dir) if fname.endswith(".jpg")
                              ])
    target_img_paths = sorted([os.path.join(target_dir, fname)
                               for fname in os.listdir(target_dir)
                               if fname.endswith(".png") and not fname.startswith(".")
                               ])
    print("Number of images/mask", len(input_img_paths))

    print("[INFO] loading images...")

    # Split our img paths into a training and a validation set
    val_samples = 1000
    # import random
    # random.Random(1337).shuffle(input_img_paths)
    # random.Random(1337).shuffle(target_img_paths)
    train_image_list = input_img_paths[:-val_samples]
    train_mask_list = target_img_paths[:-val_samples]
    val_img_list = input_img_paths[-val_samples:]
    val_mask_list = target_img_paths[-val_samples:]
    return train_image_list,train_mask_list,val_img_list,val_mask_list

def get_train_gen(IMG_SIZE, BATCH_SIZE):
    train_image_list, train_mask_list, _,_ = get_train_test_split()
    # define image augmentation operations for train set
    aug_train = albu.Compose([
        albu.Blur(blur_limit=3),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.3, brightness_by_max=True),
        # albu.RandomGamma(),
        albu.augmentations.transforms.Resize(height=IMG_SIZE, width=IMG_SIZE),
        albu.RandomSizedCrop((IMG_SIZE - 50, IMG_SIZE - 1), IMG_SIZE, IMG_SIZE)
    ])
    # construct train and test data generators
    train_generator = OxfordPetsData(
        input_img_paths=train_image_list,
        input_mask_paths=train_mask_list,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augmentation=aug_train)
    steps_ep = len(train_image_list)//BATCH_SIZE
    return steps_ep, train_generator

def get_test_gen(IMG_SIZE, BATCH_SIZE):
    train_image_list, train_mask_list, val_img_list, val_mask_list = get_train_test_split()
    aug_test = albu.Compose([
        albu.augmentations.transforms.Resize(height=IMG_SIZE, width=IMG_SIZE)
    ])

    # construct test data generators

    test_generator = OxfordPetsData(
        input_img_paths=val_img_list,
        input_mask_paths=val_mask_list,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augmentation=aug_test)
    return test_generator

def disp_scaled(batch_images, batch_masks,figsize=(15,15)):
    nrow = 4
    ncol = 2
    fig = plt.figure(4, figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol, #width_ratios=[1, 1, 1],
             wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)
    for i in range(nrow):
        for j in range(ncol):
            ax= plt.subplot(gs[i,j])
            if j%2==0:
                im = batch_images[i]
            else:
                im = np.squeeze(batch_masks[i], axis=-1).astype("uint8")
            ax.imshow(im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    #plt.tight_layout() # do not use this!!
    plt.show()
if __name__== "__main__":
    print("Inside if: calling Main: called from {}".format(__name__))
    main()
else:
    print("Inside Else: called from {}".format(__name__))


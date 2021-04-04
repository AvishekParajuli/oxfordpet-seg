#predict_unet2.py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from model import *
from data_loader import get_test_gen
import keras

IMG_SIZE = 160
img_size = (IMG_SIZE, IMG_SIZE)
mask_size = img_size #+(1,)#generator gives mask of size(N,H,W,1)
NUM_CLASSES = 3
BATCH_SIZE = 32


#load the model
model = unet(num_classes=3, input_size=(IMG_SIZE, IMG_SIZE, 3), rate =0.25)
model.summary()

test_generator = get_test_gen(IMG_SIZE, BATCH_SIZE)
batch_images,batch_masks = test_generator[0]

#Load the model and the pretrained weights
modelname = "unet2_{}p0_softmax.h5".format(IMG_SIZE)
model = keras.models.load_model("C:/Users/parajav/PycharmProjects/oxprj/"+modelname)

#start predicting on the batch of images
val_preds = model.predict(batch_images)
pred_mask = np.argmax(val_preds, axis=-1)


def disp_scaled(batch_images, batch_masks,pred_mask,nrow =4,offset=0, figsize=(15,15)):
    #nrow = 4
    ncol = 3
    fig = plt.figure(4, figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol, #width_ratios=[1, 1, 1],
             wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.05, right=0.845)
    for i in range(nrow):
        for j in range(ncol):
            ax= plt.subplot(gs[i,j])
            if j==0:
                im = batch_images[i+offset]
                title = "Image"
            elif j==1:
                im = np.squeeze(batch_masks[i+offset], axis=-1).astype("uint8")
                title = "True mask"
            else:
                im = pred_mask[i+offset]
                title = "pred mask"
            if i==0:
                ax.set_title(title)
            ax.imshow(im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    #plt.tight_layout() # do not use this!!
    plt.show()

disp_scaled(batch_images, batch_masks,pred_mask,nrow =4,offset=0)


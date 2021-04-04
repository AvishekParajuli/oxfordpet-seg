#train_unet2.py
import matplotlib
#matplotlib.use("Agg")# to save figure(NO UI backed)
import matplotlib.pyplot as plt
#matplotlib.use( 'tkagg' ) for UI

from model import *
from data_loader import get_test_gen, get_train_gen
import keras
from keras.callbacks import TensorBoard

IMG_SIZE = 160
img_size = (IMG_SIZE, IMG_SIZE)
mask_size = img_size #+(1,)#generator gives mask of size(N,H,W,1)
NUM_CLASSES = 3
BATCH_SIZE = 32

#keras.backend.clear_session()# free up RAM for multiple calls
#load the model
model = unet(num_classes=3, input_size=(IMG_SIZE, IMG_SIZE, 3), rate =0.25)
model.summary()

val_samples = 1000
# construct train and test data generators
steps_ep, train_generator = get_train_gen(IMG_SIZE, BATCH_SIZE)

test_generator = get_test_gen(IMG_SIZE, BATCH_SIZE)
batch_images,batch_masks =test_generator[0]


# we start here
modelname = "unet2_{}p0_softmax.h5".format(IMG_SIZE)

tb = TensorBoard(histogram_freq=1,write_images= True, log_dir='logs', write_graph=True)
mc = keras.callbacks.ModelCheckpoint(mode='max', filepath=modelname, monitor='acc', save_best_only='True', verbose=1)
es = keras.callbacks.EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)
callbacks = [tb, mc, es]
#opt = keras.optimizers.adam( lr= 0.0001 , decay = 0,  clipnorm = 0.5 )
#lossfunc = tensorflow.keras.losses.sparse_categorical_crossentropy(from_logits=True)
#model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
#model.compile(loss=lossfunc, optimizer='adam',metrics=["accuracy"])
EPOCHS = 100

H = model.fit(train_generator, epochs=EPOCHS,
              steps_per_epoch=steps_ep,
              validation_data=(batch_images,batch_masks),#workaround to pass only the batch data (tensorboard, histgram_freq>0 requirement)
              validation_steps=val_samples//BATCH_SIZE,
              callbacks=callbacks)
import pickle
with open("history", 'wb') as f:
    pickle.dump(H, f)

"""epochs = 40
H = model.fit_generator(train_generator, epochs=epochs,
                        steps_per_epoch=len(train_input_img_paths) // BATCH_SIZE,
                        validation_data=test_generator,
                        validation_steps=len(val_input_img_paths) // BATCH_SIZE,
                        callbacks=callbacks)
"""
#model.save("C:/Users/parajav/PycharmProjects/oxprj/keras_Aug_256p0_drop_softmax_final.h5")
#val_preds = model.predict(val_gen)
'''
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./accFigure1.png")
'''

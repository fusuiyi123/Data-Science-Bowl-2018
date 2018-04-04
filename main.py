from data_utils import read_train_data,read_test_data,prob_to_rles,mask_to_rle,resize,np
from model import get_unet
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint


batch_size = 32
epochs = 5


# get train_data
train_img, train_mask = read_train_data()

# get test_data
test_img, test_img_sizes = read_test_data()

# get u_net model
u_net = get_unet()

# fit model on train_data
print("\nTraining...")
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

result = u_net.fit(train_img, train_mask, batch_size=batch_size, epochs=epochs, validation_split=0.2)

print("Predicting")
# Predict on test data
test_mask = u_net.predict(test_img, verbose=1)

# Create list of upsampled test masks
test_mask_upsampled = []
for i in range(len(test_mask)):
    test_mask_upsampled.append(resize(np.squeeze(test_mask[i]),
                                       (test_img_sizes[i][0],test_img_sizes[i][1]),
                                       mode='constant', preserve_range=True))

test_ids,rles = mask_to_rle(test_mask_upsampled)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018.csv', index=False)

print("Data saved")

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation

# Load a single image and its associated masks
id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
file = "stage1_train/{}/images/{}.png".format(id,id)
masks = "stage1_train/{}/masks/*.png".format(id)
image = skimage.io.imread(file)
masks = skimage.io.imread_collection(masks).concatenate()
height, width, _ = image.shape
num_masks = masks.shape[0]
#!/usr/bin/env python
# coding: utf-8

# In[13]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[14]:


import pandas as pd
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
import re

from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from sklearn.model_selection import train_test_split


# In[15]:


calc_train = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_train_set.csv')
calc_test = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_test_set.csv')
mass_train = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv')
mass_test = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv')
dicom_data = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv')


# In[16]:


image_dir = '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/jpeg/'
full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
roi_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path

full_mammogram_images = full_mammogram_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
cropped_images = cropped_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
roi_mask_images = roi_mask_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
full_mammogram_images.head()


# In[17]:


dicom_cleaning_data = dicom_data.copy()
dicom_cleaning_data['image_path'] = dicom_cleaning_data['image_path'].str.replace('CBIS-DDSM/jpeg/', image_dir)


# In[18]:


dicom_cleaning_data.drop(['PatientBirthDate','AccessionNumber','Columns','ContentDate','ContentTime','PatientSex','PatientBirthDate',
                                                'ReferringPhysicianName','Rows','SOPClassUID','SOPInstanceUID',
                                                'StudyDate','StudyID','StudyInstanceUID','StudyTime','InstanceNumber','SeriesInstanceUID','SeriesNumber'],axis =1, inplace=True)
dicom_cleaning_data.info()


# In[19]:


dicom_cleaning_data['SeriesDescription'].fillna(0, axis = 0, inplace=True)
dicom_cleaning_data['Laterality'].fillna(method = 'bfill', axis = 0, inplace=True)

dicom_cleaning_data.isna().sum()


# In[20]:


label_mapping = {'BENIGN': 0, 'MALIGNANT': 1, 'BENIGN_WITHOUT_CALLBACK': 2}
calc_train['label'] = calc_train['pathology'].map(label_mapping)
calc_test['label'] = calc_test['pathology'].map(label_mapping)
mass_train['label'] = mass_train['pathology'].map(label_mapping)
mass_test['label'] = mass_test['pathology'].map(label_mapping)


# In[21]:


dicom_model = dicom_data.copy()
dicom_model['image_path'] = dicom_cleaning_data['image_path'].str.replace('CBIS-DDSM/jpeg/', image_dir)

def load_and_process_image(image_path):
    image = load_img(image_path, target_size=(224,224), color_mode="grayscale")
    image = img_to_array(image) / 255.0
    return image
    
def match1(file_path):
    patientID = file_path.split('/')[0]
    series_description = 'full mammogram images'
    filtered_df = dicom_cleaning_data[(dicom_cleaning_data['SeriesDescription'] == series_description) & 
                            (dicom_cleaning_data['PatientName'] == patientID)]

    if filtered_df.empty:
        return None
    #print(1)
    return filtered_df['image_path'].iloc[0]

def match2(file_path):
    patientID = file_path.split('/')[0]
    series_description = 'cropped images'
    filtered_df = dicom_cleaning_data[(dicom_cleaning_data['SeriesDescription'] == series_description) & 
                            (dicom_cleaning_data['PatientName'] == patientID)]
    if filtered_df.empty:
        return None
    #print(2)
    return filtered_df['image_path'].iloc[0]

def match3(file_path):
    patientID = file_path.split('/')[0]
    series_description = 'ROI mask images'

    filtered_df = dicom_cleaning_data[(dicom_cleaning_data['SeriesDescription'] == series_description) & 
                            (dicom_cleaning_data['PatientName'] == patientID)]
    if filtered_df.empty:
        print('no')
        return None
    #print(3)
    return filtered_df['image_path'].iloc[0]

def load_data(df):
    full_imgs = []
    cropped_imgs = []
    roi_imgs = []
    labels = []
    for _, row in df.iterrows():
        full_img_path = match1(row['image file path'])
        if full_img_path is None:
            continue
        cropped_img_path = match2(row['cropped image file path'])
        if cropped_img_path is None:
            continue
        roi_img_path = match3(row['ROI mask file path'])
        if roi_img_path is None:
            continue
        if full_img_path is not None and cropped_img_path is not None and roi_img_path is not None:
            if os.path.exists(full_img_path) and os.path.exists(cropped_img_path) and os.path.exists(roi_img_path):
                    full_imgs.append(load_and_process_image(full_img_path))
                    cropped_imgs.append(load_and_process_image(cropped_img_path))
                    roi_imgs.append(load_and_process_image(roi_img_path))
                    labels.append(row['label'])
            

    return np.array(full_imgs), np.array(cropped_imgs), np.array(roi_imgs), np.array(labels)


# In[22]:


calc_train['image file path'].nunique()
calc_train_model = calc_train.copy()
calc_train_model = calc_train_model.drop_duplicates(subset=['image file path']).reset_index(drop=True)
calc_train_model['image file path'].nunique()


# In[23]:


print(mass_train['image file path'].nunique())
mass_train_model = mass_train.copy()
mass_train_model = mass_train_model.drop_duplicates(subset=['image file path']).reset_index(drop=True)
mass_train_model['image file path'].nunique()


# In[24]:


print(mass_test['image file path'].nunique())
mass_test_model = mass_test.copy()
mass_test_model = mass_test_model.drop_duplicates(subset=['image file path']).reset_index(drop=True)
mass_test_model['image file path'].nunique()


# In[25]:


print(calc_test['image file path'].nunique())
calc_test_model = calc_test.copy()
calc_test_model = calc_test_model.drop_duplicates(subset=['image file path']).reset_index(drop=True)
calc_test_model['image file path'].nunique()


# In[26]:


calc_train_model.info()


# In[27]:


print(match1(calc_train_model['image file path'][1000]))
print(match2(calc_train_model['cropped image file path'][1000]))
print(match3(calc_train_model['ROI mask file path'][1000]))
calc_train_model['label'][1000]


# In[28]:


x_calc_full_train, x_calc_cropped_train, x_calc_roi_train, y_calc_train = [],[],[],[]
x_calc_full_train, x_calc_cropped_train, x_calc_roi_train, y_calc_train = load_data(calc_train_model)

x_calc_full_train.shape


# In[29]:


x_calc_full_test = x_calc_full_train[1000:]
x_calc_cropped_test = x_calc_cropped_train[1000:]
x_calc_roi_test = x_calc_roi_train[1000:]
y_calc_test = y_calc_train[1000:]

x_calc_full_train = x_calc_full_train[:1000]
x_calc_cropped_train = x_calc_cropped_train[:1000]
x_calc_roi_train = x_calc_roi_train[:1000]
y_calc_train = y_calc_train[:1000]


# In[30]:


x_calc_roi_train.shape


# In[31]:


x_mass_full_train, x_mass_cropped_train, x_mass_roi_train, y_mass_train = [],[],[],[]
x_mass_full_train, x_mass_cropped_train, x_mass_roi_train, y_mass_train = load_data(mass_train_model)
x_mass_cropped_train.shape


# In[32]:


x_mass_full_test, x_mass_cropped_test, x_mass_roi_test, y_mass_test = [], [], [], []
x_mass_full_test, x_mass_cropped_test, x_mass_roi_test, y_mass_test = load_data(mass_test_model)
x_mass_cropped_test.shape


# In[33]:


x_full = np.concatenate([x_calc_full_train,x_mass_full_train], axis=0)
x_cropped = np.concatenate([x_calc_cropped_train,x_mass_cropped_train], axis=0)
x_roi = np.concatenate([x_calc_roi_train,x_mass_roi_train], axis=0)
y = np.concatenate([y_calc_train,y_mass_train], axis=0)

# Combining testing data.
x_full_test = np.concatenate([x_calc_full_test,x_mass_full_test], axis=0)
x_cropped_test = np.concatenate([x_calc_cropped_test,x_mass_cropped_test], axis=0)
x_roi_test = np.concatenate([x_calc_roi_test,x_mass_roi_test], axis=0)
y_test = np.concatenate([y_calc_test,y_mass_test], axis=0)

print(x_full.shape, x_cropped.shape, x_roi.shape, y.shape)
print(x_full_test.shape, x_cropped_test.shape, x_roi_test.shape, y_test.shape)


# In[34]:


from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)


# In[35]:


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model

def UNet3Plus(input_shape):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Decoder
    u2 = UpSampling2D((2, 2))(c2)
    u2 = concatenate([u2, c1], axis=-1)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c3 = BatchNormalization()(c3)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c3)

    model = Model(inputs, outputs)
    return model

# Example usage
input_shape = (224, 224, 1)  # Grayscale image
model = UNet3Plus(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Ensure labels are segmentation masks
x_train, x_val, y_train, y_val = train_test_split(x_roi, x_roi, test_size=0.2, random_state=42)

# Verify shapes
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

# Compile model with lower learning rate and gradient clipping
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('unet3plus_best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stopping, lr_scheduler]
)


# In[ ]:


from sklearn.mixture import GaussianMixture

def apply_gmm_to_segmentation(segmented_images):
    refined_masks = []
    for image in segmented_images:
        gmm = GaussianMixture(n_components=2)
        image_reshaped = image.reshape(-1, 1)
        gmm.fit(image_reshaped)
        labels = gmm.predict(image_reshaped)
        refined_mask = labels.reshape(image.shape)
        refined_masks.append(refined_mask)
    return np.array(refined_masks)

# Example usage
segmented_test_masks = model.predict(x_roi_test)
refined_test_masks = apply_gmm_to_segmentation(segmented_test_masks)


# In[ ]:


# Use ROI masks as ground truth
y_test = x_roi_test

# Predict and binarize predictions
y_pred = model.predict(x_roi_test)
y_pred = (y_pred > 0.5).astype(np.float32)

# Ensure shapes match
print(f"x_roi_test shape: {x_roi_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"y_pred shape: {y_pred.shape}")

# Dice Coefficient Function
def dice_coefficient(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    intersection = np.sum(y_true_flat * y_pred_flat)
    return (2. * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat))

# Calculate Dice Coefficient
dice = dice_coefficient(y_test, y_pred)
print(f"Dice Coefficient: {dice}")


# In[ ]:


import matplotlib.pyplot as plt

def visualize_segmentation(image, mask, refined_mask):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image.squeeze(), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("Segmented Mask")
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Refined Mask (GMM)")
    plt.imshow(refined_mask.squeeze(), cmap='gray')
    plt.show()

# Example
visualize_segmentation(x_roi_test[0], segmented_test_masks[0], refined_test_masks[0])

visualize_segmentation(x_roi_test[1], segmented_test_masks[1], refined_test_masks[1])



# In[ ]:


import matplotlib.pyplot as plt
# Extract training and validation accuracy
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


# Extract training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


def test_model(test_images, test_masks):
    y_pred = model.predict(test_images)
    y_pred = (y_pred > 0.5).astype(np.float32)  # Binarize predictions

    # Dice Coefficient
    def dice_coefficient(y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        intersection = np.sum(y_true_flat * y_pred_flat)
        return (2. * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat))

    # Calculate Dice for each image
    dice_scores = []
    for i in range(len(test_images)):
        dice = dice_coefficient(test_masks[i], y_pred[i])
        dice_scores.append(dice)

    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice Coefficient: {mean_dice}")
    return dice_scores

# Testing on validation set as an example
dice_scores = test_model(x_val, y_val)

# Visualize test results
for i in range(10):  # Show 5 test examples
    visualize_segmentation(x_val[i], y_val[i], y_pred[i])


# In[ ]:





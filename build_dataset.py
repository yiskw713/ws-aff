import glob
import pandas as pd
import numpy as np
import scipy.io

dir_path = glob.glob('./part-affordance-dataset/tools/*')


object_to_cls = {
    'background': 0,
    'bowl': 1,
    'cup': 2,
    'hammer': 3,
    'knife': 4,
    'ladle': 5,
    'mallet': 6,
    'mug': 7,
    'pot': 8,
    'saw': 9,
    'scissors': 10,
    'scoop': 11,
    'shears': 12,
    'shovel': 13,
    'spoon': 14,
    'tenderizer': 15,
    'trowel': 16,
    'turner': 17
}


image_path = []
image_level_aff_path = []
image_level_obj_path = []
pixel_level_aff_path = []


for d in dir_path:
    img_path = glob.glob(d + '/*.jpg')
    o = [0, object_to_cls[d[32:-3]]]    # path[32:-3] => object name
    obj_multi_hot = np.zeros(18, dtype=np.int64)
    obj_multi_hot[o] = 1

    for img in img_path:
        aff_multi_hot = np.zeros(8, dtype=np.int64)
        pix_lev_aff_path = img[:-7] + 'label.mat'
        label = scipy.io.loadmat(pix_lev_aff_path)['gt_label']
        for i in range(0, 8):
            if i in label:
                aff_multi_hot[i] = 1

        image_path.append(img)
        image_level_aff_path.append(img[:-7] + 'aff.npy')
        image_level_obj_path.append(img[:-7] + 'obj.npy')
        pixel_level_aff_path.append(pix_lev_aff_path)
        np.save(img[:-7] + 'obj.npy', obj_multi_hot)
        np.save(img[:-7] + 'aff.npy', aff_multi_hot)


image_train = []
image_test = []
image_level_aff_train = []
image_level_aff_test = []
obj_train = []
obj_test = []
pixel_level_aff_train = []
pixel_level_aff_test = []

for i, (img, img_lev_aff, obj, pix_lev_aff) \
        in enumerate(zip(image_path, image_level_aff_path, image_level_obj_path, pixel_level_aff_path)):

    if i % 5 == 0:
        image_test.append(img)
        image_level_aff_test.append(img_lev_aff)
        obj_test.append(obj)
        pixel_level_aff_test.append(pix_lev_aff)
    else:
        image_train.append(img)
        image_level_aff_train.append(img_lev_aff)
        obj_train.append(obj)
        pixel_level_aff_train.append(pix_lev_aff)


df_train = pd.DataFrame({
    'image': image_train,
    'image_level_affordance': image_level_aff_train,
    'object': obj_train,
    'pixel_level_affordance': pixel_level_aff_train},
    columns=['image', 'image_level_affordance',
             'object', 'pixel_level_affordance']
)


df_test = pd.DataFrame({
    'image': image_test,
    'image_level_affordance': image_level_aff_test,
    'object': obj_test,
    'pixel_level_affordance': pixel_level_aff_test},
    columns=['image', 'image_level_affordance',
             'object', 'pixel_level_affordance']
)


data = pd.concat([df_train, df_test])


df_train.to_csv('./part-affordance-dataset/train.csv', index=None)
df_test.to_csv('./part-affordance-dataset/test.csv', index=None)
data.to_csv('./part-affordance-dataset/all_data.csv', index=None)

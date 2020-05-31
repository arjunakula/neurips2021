from scipy.ndimage.interpolation import zoom
import matplotlib.cm as mpl_color_map
import os
import cv2
import pickle
import numpy as np
import copy
from PIL import Image, ImageFilter


def apply_colormap_on_image(org_im, activation, colormap_name='hsv'):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image



folder_path = '/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/'

# img = cv2.imread(os.path.join(folder_path, 'CLEVR_val_000000.png'))
img = Image.open(os.path.join(folder_path, 'data/clevr_ref+_1.0/images/val/CLEVR_val_000000.png'))
# cv2.imshow("haha", img)
# cv2.waitKey(0)

fid = open(os.path.join(folder_path, 'feats.pkl'), 'rb')
feats_data = pickle.load(fid)
fid.close()

# for feats_i in feats_data:
#     # print(feats_i.shape)
#     print(feats_i['feats'].numpy().shape)
#     feats_data_np = feats_i['feats'].numpy()

# for i in range(feats_data_np.shape[0]):
#     feats_data_np_i = feats_data_np[i]

#     cam = np.maximum(feats_data_np_i, 0)
#     cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
#     cam = cv2.resize(cam, (img.size[0], img.size[1]))
#     cam = np.uint8(cam * 255) 
#     # cam = zoom(cam, np.array(img.shape)/np.array(cam.shape))

#     _, heatmap_on_image = apply_colormap_on_image(img, cam)

#     heatmap_on_image.save(os.path.join("C:\\Users\\kezew\\Downloads\\imgs\\", str(i) + '.png'))
#     # cv2.imwrite( os.path.join("C:\\Users\\kezew\\Downloads\\imgs\\", str(i) + '.png'), heatmap_on_image )
#     # cv2.imshow("haha", cam)
#     # cv2.waitKey(0)


fid = open(os.path.join(folder_path, 'filter.pkl'), 'rb')
filter_data = pickle.load(fid)ls
fid.close()

i = 0
for filter_i in filter_data:
    print(filter_i['max_filter'].shape)
    # print(filter_i[].shape)

    filter_i_np = filter_i['max_filter']

    cam = np.maximum(filter_i_np, 0)
    #cam = filter_i_np
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    cam = np.uint8(cam * 255) 
    # cam = zoom(cam, np.array(img.shape)/np.array(cam.shape))

    _, heatmap_on_image = apply_colormap_on_image(img, cam)

    heatmap_on_image.save(os.path.join(folder_path, str(i) + '.png'))

    i += 1





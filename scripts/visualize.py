from scipy.ndimage.interpolation import zoom
import matplotlib.cm as mpl_color_map
import os
import cv2
import pickle
import numpy as np
import copy
from PIL import Image, ImageFilter
import random


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
img = Image.open(os.path.join(folder_path, 'data/clevr_ref+_1.0/images/val/CLEVR_val_000041.png'))
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
filter_data = pickle.load(fid)
fid.close()

i = 0
for filter_i in filter_data:
    i = i +1
    if(i <= 3 or i >4):
        continue
    print(filter_i['max_filter'].shape)
    # print(filter_i[].shape)

    
    filter_i_np = filter_i['max_filter']

    k1 = 0.1
    k2 = 0.0
    k3 = 0.2
    k4 = 1.0
    k5 = 0.1
    k6 = 0.0
    k7 = 1.1
    k8 = 0.1
    filter_i_np = np.zeros_like(filter_i['max_filter'])
    for n in range(0,3):
        for m in range(0,5):
            filter_i_np[m][n] = k1 + random.random()+ random.random()
        for n in range(5,10):
            filter_i_np[m][n] = k1 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k1 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k1 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k1 + random.random()+ random.random()

    for n in range(3,5):
        for m in range(0,5):
            filter_i_np[m][n] = k2 + random.random()+ random.random()
        for n in range(5,10):
            filter_i_np[m][n] = k2 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k2 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k2 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k2 + random.random()+ random.random()


    for n in range(5,8):
        for m in range(0,5):
            filter_i_np[m][n] = k3 + random.random()+ random.random()
        for n in range(5,10):
            filter_i_np[m][n] = k3 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k3 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k3 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k3 + random.random()+ random.random()

    for n in range(8,10):
        for m in range(0,5):
            filter_i_np[m][n] = k4 + random.random()+ random.random()
        for n in range(5,10):
            filter_i_np[m][n] = k4 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k4 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k4 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k4 + random.random()+ random.random()

    for n in range(10,13):
        for m in range(0,5):
            filter_i_np[m][n] = k5 + random.random()+ random.random()
        for n in range(5,10):
            filter_i_np[m][n] = k5 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k5 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k5 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k5 + random.random()+ random.random()

    for n in range(13,15):
        for m in range(0,5):
            filter_i_np[m][n] = k6 + random.random()+ random.random()
        for n in range(5,10):
            filter_i_np[m][n] = k6 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k6 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k6 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k6 + random.random()+ random.random()
   
    for n in range(15,18):
        for m in range(0,5):
            filter_i_np[m][n] = k7 + random.random()+ random.random()
        for n in range(5,10):
            filter_i_np[m][n] = k7 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k7 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k7 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k7 + random.random()+ random.random()

    for n in range(15,20):
        for m in range(0,5):
            filter_i_np[m][n] = k8 + random.random() + random.random()
        for n in range(5,10):
            filter_i_np[m][n] = k8 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k8 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k8 + random.random()+ random.random()
        for n in range(10,15):
            filter_i_np[m][n] = k8 + random.random()+ random.random()

    filter_i_np = filter_i['max_filter']

    #image 1_1 if(i <= 3 or i >4):

    # filter_i_np[4][4]=filter_i_np[4][4]-2.0
    # filter_i_np[4][5]=filter_i_np[4][5]-2.0
    # filter_i_np[4][6]=filter_i_np[4][6]-2.0
    # filter_i_np[6][14]=filter_i_np[6][14]+2.0
    # filter_i_np[6][15]=filter_i_np[6][15]+2.0
    # filter_i_np[6][6]=filter_i_np[6][6]+1.3
    # filter_i_np[6][7]=filter_i_np[6][7]+1.2
    
    # filter_i_np[5][3]=filter_i_np[5][3]-2.1
    # filter_i_np[5][4]=filter_i_np[5][4]-2.2
    # filter_i_np[5][5]=filter_i_np[5][5]-1.89
    # filter_i_np[5][6]=filter_i_np[5][6]-2.1
    # filter_i_np[6][3]=filter_i_np[6][3]-2.0
    # filter_i_np[6][4]=filter_i_np[6][4]-2.2
    # filter_i_np[6][5]=filter_i_np[6][5]+0.2
    # filter_i_np[6][6]=filter_i_np[6][6]+0.44
    # filter_i_np[6][7]=filter_i_np[6][7]-0.73
    # filter_i_np[6][8]=filter_i_np[6][8]-2.02
    # for k in range(9,19):
    #     filter_i_np[6][k] = filter_i_np[6][k] - 1.41
    #     filter_i_np[5][k] = filter_i_np[5][k] - 0.91
    # for k in range(13,15):
    #     filter_i_np[7][k] = filter_i_np[7][k] + 0.82
    #     filter_i_np[9][k] = filter_i_np[9][k] + 0.893
    # for k in range(15,18):
    #     filter_i_np[10][k] = filter_i_np[10][k] + 2.968
    #     filter_i_np[11][k] = filter_i_np[11][k] + 2.02

    # for k in range(3,10):
    #     filter_i_np[8][k] = filter_i_np[8][k] - 4.12
    #     filter_i_np[9][k] = filter_i_np[9][k] - 3.16
    #     filter_i_np[10][k] = filter_i_np[10][k] - 4.72
    #     filter_i_np[11][k] = filter_i_np[11][k] - 5.01
    #     filter_i_np[12][k] = filter_i_np[12][k] - 5.05


    # filter_i_np[7][3]=filter_i_np[7][3]-4.17
    # filter_i_np[7][4]=filter_i_np[7][4]-4.12
    # filter_i_np[7][5]=filter_i_np[7][5]-4.21
    # filter_i_np[7][6]=filter_i_np[7][6]-4.14


    # fp2 = open("/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/ex2_row3_column3.pkl","rb")
    # filter_i_np = pickle.load(fp2)[0]['max_filter']
    # fp2.close()


    # filter_i_np[4][5]=filter_i_np[4][5]+1.0
    # filter_i_np[4][6]=filter_i_np[4][6]+1.0
    # filter_i_np[6][6]=filter_i_np[6][6]+1.3
    # filter_i_np[6][7]=filter_i_np[6][7]+1.2
   

    # filter_i_np[3][3]=filter_i_np[3][4]+1.3
    # filter_i_np[3][4]=filter_i_np[3][5]+1.2
    # filter_i_np[3][9]=filter_i_np[3][9]+1.3
    # filter_i_np[3][10]=filter_i_np[3][10]+1.2

    # filter_i_np[6][12]=filter_i_np[6][12]+1.3
    # filter_i_np[6][13]=filter_i_np[6][13]+1.2

    # filter_i_np[11][4]=filter_i_np[11][4]+2.1
    # filter_i_np[11][5]=filter_i_np[11][5]+2.1

    # filter_i_np[13][7]=filter_i_np[13][7]+1.1
    # filter_i_np[13][8]=filter_i_np[13][8]+0.8



    # image 1_2
    filter_i_np = filter_i['max_filter']
    filter_i_np[4][5]=filter_i_np[4][5]+1.3
    filter_i_np[4][6]=filter_i_np[4][6]+0.7
    filter_i_np[6][6]=filter_i_np[6][6]+0.9
    filter_i_np[6][7]=filter_i_np[6][7]+0.95
    filter_i_np[6][8]=filter_i_np[6][8]-0.1
    filter_i_np[6][9]=filter_i_np[6][9]-0.25

    filter_i_np[3][9]=filter_i_np[3][9]+1.15
    filter_i_np[3][10]=filter_i_np[3][10]+0.5

    filter_i_np[6][12]=filter_i_np[6][12]+1.0
    filter_i_np[6][13]=filter_i_np[6][13]+1.0

    filter_i_np[11][4]=filter_i_np[11][4]+2.0
    filter_i_np[11][5]=filter_i_np[11][5]+2.0
    filter_i_np[11][6]=filter_i_np[11][6]+1.0
    filter_i_np[11][7]=filter_i_np[11][7]+0.2

    filter_i_np[13][7]=filter_i_np[13][7]+0.7
    filter_i_np[13][8]=filter_i_np[13][8]+0.8

    # #image 2_1 if(i <= 4 or i >5):
    # filter_i_np = filter_i['max_filter']
    # filter_i_np[4][5]=filter_i_np[4][5]-7.0
    # filter_i_np[4][6]=filter_i_np[4][6]-4.9
    # filter_i_np[6][6]=filter_i_np[6][6]-4.7
    # filter_i_np[6][7]=filter_i_np[6][7]-4.75
    # filter_i_np[6][8]=filter_i_np[6][8]-4.2
    # filter_i_np[6][9]=filter_i_np[6][9]-4.2
    # filter_i_np[6][10]=filter_i_np[6][8]-4.2
    # filter_i_np[6][11]=filter_i_np[6][9]-4.2
    # filter_i_np[6][12]=filter_i_np[6][8]-4.2
    # filter_i_np[6][13]=filter_i_np[6][9]-4.2
    # filter_i_np[6][14]=filter_i_np[6][8]-4.2
    # filter_i_np[6][15]=filter_i_np[6][9]-4.2

    # filter_i_np[7][6]=filter_i_np[6][6]-1.7
    # filter_i_np[7][7]=filter_i_np[6][7]-1.75
    # filter_i_np[7][8]=filter_i_np[6][8]-1.2
    # filter_i_np[7][9]=filter_i_np[6][9]-1.2
    # filter_i_np[7][10]=filter_i_np[6][8]-1.2
    # filter_i_np[7][11]=filter_i_np[6][9]-1.2
    # filter_i_np[7][12]=filter_i_np[6][8]-1.2
    # filter_i_np[7][13]=filter_i_np[6][9]-1.2
    # filter_i_np[7][14]=filter_i_np[6][8]-1.2
    # filter_i_np[7][15]=filter_i_np[6][9]-1.2

    
    # filter_i_np[6][12]=filter_i_np[6][12]-7.4
    # filter_i_np[6][13]=filter_i_np[6][13]-7.2

    # filter_i_np[8][4]=filter_i_np[9][4]-9.2
    # filter_i_np[8][5]=filter_i_np[9][5]-7.2
    # filter_i_np[8][6]=filter_i_np[9][6]-4.0
    # filter_i_np[8][7]=filter_i_np[9][7]-4.3
    # filter_i_np[8][8]=filter_i_np[9][8]-4.3
    # filter_i_np[8][9]=filter_i_np[9][9]-4.3
    # filter_i_np[8][10]=filter_i_np[9][10]-4.3
    # filter_i_np[8][11]=filter_i_np[9][11]-4.3
    # filter_i_np[8][12]=filter_i_np[9][8]-4.3
    # filter_i_np[8][13]=filter_i_np[9][9]-4.3
    # filter_i_np[8][14]=filter_i_np[9][10]-4.3
    # filter_i_np[8][15]=filter_i_np[9][11]-4.3

    # filter_i_np[9][4]=filter_i_np[9][4]-9.2
    # filter_i_np[9][5]=filter_i_np[9][5]-7.2
    # filter_i_np[9][6]=filter_i_np[9][6]-4.0
    # filter_i_np[9][7]=filter_i_np[9][7]-4.3
    # filter_i_np[9][8]=filter_i_np[9][6]-4.0
    # filter_i_np[9][9]=filter_i_np[9][7]-4.3
    # filter_i_np[9][10]=filter_i_np[9][6]-4.0
    # filter_i_np[9][11]=filter_i_np[9][7]-4.3

    # filter_i_np[10][1]=filter_i_np[10][4]-6.2
    # filter_i_np[10][2]=filter_i_np[10][5]-5.2
    # filter_i_np[10][3]=filter_i_np[10][6]-8.0
    # filter_i_np[10][4]=filter_i_np[10][4]-6.2
    # filter_i_np[10][5]=filter_i_np[10][5]-5.2
    # filter_i_np[10][6]=filter_i_np[10][6]-8.0
    # filter_i_np[10][7]=filter_i_np[10][7]-2.3
    # filter_i_np[10][8]=filter_i_np[9][6]-4.0
    # filter_i_np[10][9]=filter_i_np[9][7]-4.3
    # filter_i_np[10][10]=filter_i_np[9][6]-4.0
    # filter_i_np[10][11]=filter_i_np[9][7]-4.3
    # filter_i_np[10][12]=filter_i_np[9][6]-4.0
    # filter_i_np[10][13]=filter_i_np[9][7]-4.3
    # filter_i_np[10][14]=filter_i_np[9][6]-4.0
    # filter_i_np[10][15]=filter_i_np[9][7]-4.3

    # filter_i_np[11][1]=filter_i_np[11][4]-14.2
    # filter_i_np[11][2]=filter_i_np[11][5]-5.2
    # filter_i_np[11][3]=filter_i_np[11][6]-6.0
    # filter_i_np[11][4]=filter_i_np[11][4]+3.2
    # filter_i_np[11][5]=filter_i_np[11][5]+3.2
    # filter_i_np[11][6]=filter_i_np[11][6]-6.0
    # filter_i_np[11][7]=filter_i_np[11][7]-8.3
    # filter_i_np[11][8]=filter_i_np[9][6]-4.0
    # filter_i_np[11][9]=filter_i_np[9][7]-4.3
    # filter_i_np[11][10]=filter_i_np[9][6]-4.0
    # filter_i_np[11][11]=filter_i_np[9][7]-4.3
    # filter_i_np[11][12]=filter_i_np[9][6]-4.0
    # filter_i_np[11][13]=filter_i_np[9][7]-4.3
    # filter_i_np[11][14]=filter_i_np[9][6]-4.0
    # filter_i_np[11][15]=filter_i_np[9][7]-4.3

    # filter_i_np[12][7]=filter_i_np[11][7]+3.5
    # filter_i_np[12][8]=filter_i_np[9][6]+3.9
    # filter_i_np[12][9]=filter_i_np[9][7]+3.0
    # filter_i_np[12][10]=filter_i_np[9][6]+1.0
    # filter_i_np[13][7]=filter_i_np[11][7]+5.3
    # filter_i_np[13][8]=filter_i_np[9][6]+4.0
    # filter_i_np[13][9]=filter_i_np[9][7]+4.3
    # filter_i_np[13][10]=filter_i_np[9][6]+4.0
 
    
    
    # filter_i_np[19][0]=filter_i_np[6][6]-1.7
    # filter_i_np[19][1]=filter_i_np[6][7]-1.75
    # filter_i_np[19][2]=filter_i_np[6][8]-1.2
    # filter_i_np[19][3]=filter_i_np[6][9]-1.2
    # filter_i_np[19][4]=filter_i_np[6][8]-1.2
    # filter_i_np[19][11]=filter_i_np[6][9]-1.2
    # filter_i_np[19][12]=filter_i_np[6][8]-1.2
    # filter_i_np[19][13]=filter_i_np[6][9]-1.2
    # filter_i_np[19][14]=filter_i_np[6][8]-1.2
    # filter_i_np[19][15]=filter_i_np[6][9]-1.2

    # #image 2_2 if(i <= 4 or i >5):
    # filter_i_np = filter_i['max_filter']
    # filter_i_np[4][5]=filter_i_np[4][5]-7.0
    # filter_i_np[4][6]=filter_i_np[4][6]-4.9
    # filter_i_np[6][6]=filter_i_np[6][6]-4.7
    # filter_i_np[6][7]=filter_i_np[6][7]-4.75
    # filter_i_np[6][8]=filter_i_np[6][8]-4.2
    # filter_i_np[6][9]=filter_i_np[6][9]-4.2
    # filter_i_np[6][10]=filter_i_np[6][8]-4.2
    # filter_i_np[6][11]=filter_i_np[6][9]-4.2
    # filter_i_np[6][12]=filter_i_np[6][8]-4.2
    # filter_i_np[6][13]=filter_i_np[6][9]-4.2
    # filter_i_np[6][14]=filter_i_np[6][8]-4.2
    # filter_i_np[6][15]=filter_i_np[6][9]-4.2

    # filter_i_np[7][6]=filter_i_np[6][6]-1.7
    # filter_i_np[7][7]=filter_i_np[6][7]-1.75
    # filter_i_np[7][8]=filter_i_np[6][8]-1.2
    # filter_i_np[7][9]=filter_i_np[6][9]-1.2
    # filter_i_np[7][10]=filter_i_np[6][8]-1.2
    # filter_i_np[7][11]=filter_i_np[6][9]-1.2
    # filter_i_np[7][12]=filter_i_np[6][8]-1.2
    # filter_i_np[7][13]=filter_i_np[6][9]-1.2
    # filter_i_np[7][14]=filter_i_np[6][8]-1.2
    # filter_i_np[7][15]=filter_i_np[6][9]-1.2

    
    # filter_i_np[6][12]=filter_i_np[6][12]-7.4
    # filter_i_np[6][13]=filter_i_np[6][13]-7.2

    # filter_i_np[8][4]=filter_i_np[9][4]-9.2
    # filter_i_np[8][5]=filter_i_np[9][5]-7.2
    # filter_i_np[8][6]=filter_i_np[9][6]-4.0
    # filter_i_np[8][7]=filter_i_np[9][7]-4.3
    # filter_i_np[8][8]=filter_i_np[9][8]-4.3
    # filter_i_np[8][9]=filter_i_np[9][9]-4.3
    # filter_i_np[8][10]=filter_i_np[9][10]-4.3
    # filter_i_np[8][11]=filter_i_np[9][11]-4.3
    # filter_i_np[8][12]=filter_i_np[9][8]-4.3
    # filter_i_np[8][13]=filter_i_np[9][9]-4.3
    # filter_i_np[8][14]=filter_i_np[9][10]-4.3
    # filter_i_np[8][15]=filter_i_np[9][11]-4.3

    # filter_i_np[9][4]=filter_i_np[9][4]-9.2
    # filter_i_np[9][5]=filter_i_np[9][5]-7.2
    # filter_i_np[9][6]=filter_i_np[9][6]-4.0
    # filter_i_np[9][7]=filter_i_np[9][7]-4.3
    # filter_i_np[9][8]=filter_i_np[9][6]-4.0
    # filter_i_np[9][9]=filter_i_np[9][7]-4.3
    # filter_i_np[9][10]=filter_i_np[9][6]-4.0
    # filter_i_np[9][11]=filter_i_np[9][7]-4.3

    # filter_i_np[10][1]=filter_i_np[9][4]-6.2
    # filter_i_np[10][2]=filter_i_np[9][5]-5.2
    # filter_i_np[10][3]=filter_i_np[9][6]-8.0
    # filter_i_np[10][4]=filter_i_np[9][4]-6.2
    # filter_i_np[10][5]=filter_i_np[9][5]-15.2
    # filter_i_np[10][6]=filter_i_np[9][6]-18.0
    # filter_i_np[10][7]=filter_i_np[9][7]-12.3
    # filter_i_np[10][8]=filter_i_np[9][6]-14.0
    # filter_i_np[10][9]=filter_i_np[9][7]-14.3
    # filter_i_np[10][10]=filter_i_np[9][6]-14.0
    # filter_i_np[10][11]=filter_i_np[9][7]-14.3
    # filter_i_np[10][12]=filter_i_np[9][6]-14.0
    # filter_i_np[10][13]=filter_i_np[9][7]-14.3
    # filter_i_np[10][14]=filter_i_np[9][6]-14.0
    # filter_i_np[10][15]=filter_i_np[9][7]-4.3

    # filter_i_np[11][1]=filter_i_np[9][4]-5.2
    # filter_i_np[11][2]=filter_i_np[9][5]-0.2
    # filter_i_np[11][3]=filter_i_np[9][6]+0.9
    # filter_i_np[11][4]=filter_i_np[9][4]+4.3
    # filter_i_np[11][5]=filter_i_np[11][5]+1.9
    # filter_i_np[11][6]=filter_i_np[9][6]+4.0
    # filter_i_np[11][7]=filter_i_np[11][7]-18.3
    # filter_i_np[11][8]=filter_i_np[9][6]-18.0
    # filter_i_np[11][9]=filter_i_np[9][7]-14.3
    # filter_i_np[11][10]=filter_i_np[9][6]-14.0
    # filter_i_np[11][11]=filter_i_np[9][7]-14.3
    # filter_i_np[11][12]=filter_i_np[9][6]-4.0
    # filter_i_np[11][13]=filter_i_np[9][7]-4.3
    # filter_i_np[11][14]=filter_i_np[9][6]-4.0
    # filter_i_np[11][15]=filter_i_np[9][7]-4.3

    # filter_i_np[12][7]=filter_i_np[9][7]+0.9
    # filter_i_np[12][8]=filter_i_np[9][6]+0.6
    # filter_i_np[12][9]=filter_i_np[9][7]+0.9
    # filter_i_np[12][10]=filter_i_np[9][6]+2.4
    # filter_i_np[13][7]=filter_i_np[11][7]+0.9
    # filter_i_np[13][8]=filter_i_np[9][6]+0.8
    # filter_i_np[13][9]=filter_i_np[9][7]+0.9
    # filter_i_np[13][10]=filter_i_np[9][6]+0.7
 
    
    
    # filter_i_np[19][0]=filter_i_np[6][6]-1.7
    # filter_i_np[19][1]=filter_i_np[6][7]-1.75
    # filter_i_np[19][2]=filter_i_np[6][8]-1.2
    # filter_i_np[19][3]=filter_i_np[6][9]-1.2
    # filter_i_np[19][4]=filter_i_np[6][8]-1.2
    # filter_i_np[19][11]=filter_i_np[6][9]-1.2
    # filter_i_np[19][12]=filter_i_np[6][8]-1.2
    # filter_i_np[19][13]=filter_i_np[6][9]-1.2
    # filter_i_np[19][14]=filter_i_np[6][8]-1.2
    # filter_i_np[19][15]=filter_i_np[6][9]-1.2
    

    import pickle
    fp3 = open("/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/ex1_row1_column1.pkl","wb")
    pickle.dump({'max_filter':filter_i_np}, fp3)
    fp3.close()

    cam = np.maximum(filter_i_np, 0)
    #cam = filter_i_np
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    cam = np.uint8(cam * 255) 
    # cam = zoom(cam, np.array(img.shape)/np.array(cam.shape))
    
    _, heatmap_on_image = apply_colormap_on_image(img, cam)

    heatmap_on_image.save(os.path.join(folder_path, str(i) + '.png'))

    





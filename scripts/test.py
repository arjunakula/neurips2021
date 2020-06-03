import pickle
import os.path
from os import path
from scipy.ndimage.interpolation import zoom
import matplotlib.cm as mpl_color_map
import os
import cv2
import pickle
import numpy as np
import copy
from PIL import Image, ImageFilter
import random
data2_filters = ['ex2_row1_column2', 'ex2_row2_column2', 'ex2_row3_column2', 'ex2_row1_column3', 'ex2_row2_column3', 'ex2_row3_column3']
data1_filters = ['ex1_row1_column2','ex1_row2_column2','ex1_row1_column3','ex1_row2_column3']
data1 = {}
data2 = {}

folder_path = '/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/'
# img = cv2.imread(os.path.join(folder_path, 'CLEVR_val_000000.png'))
img1 = Image.open(os.path.join(folder_path, 'data/clevr_ref+_1.0/images/val/CLEVR_val_000041.png'))
img2 = Image.open(os.path.join(folder_path, 'data/clevr_ref+_1.0/images/val/CLEVR_val_000129.png'))

data1['img'] = img1
data2['img'] = img2

data1['filters'] = {}
data2['filters'] = {}

for j in data1_filters:
    import pickle
    fp3 = open("/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/"+j+".pkl","rb")
    filter = pickle.load(fp3)['max_filter']
    fp3.close()
    data1['filters'][j] = filter

for j in data2_filters:
    import pickle
    fp3 = open("/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/"+j+".pkl","rb")
    filter = pickle.load(fp3)['max_filter']
    fp3.close()
    data2['filters'][j] = filter


import pickle
fp3 = open("/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/qual_example_main_paper.pkl","wb")
pickle.dump(data1, fp3)
fp3.close()

fp3 = open("/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/qual_example_appendix.pkl","wb")
pickle.dump(data2, fp3)
fp3.close()

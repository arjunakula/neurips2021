import pickle
import os.path
from os import path

fp1 = open("/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/feats.pkl","rb")
store_feats = pickle.load(fp1)

fp2 = open("/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/filter.pkl","rb")
store_filter = pickle.load(fp2)

fp3 = open("/home/arjunakula/Dropbox/My_UCLA_docs_from_2016_sept/PhD_Research/after_summer_2019/EMNLP2020_Mila_mygithub/emnlp_CL_extension/evaluation/neurips2021/test1.pkl","wb")
#pickle.dump([{'max_filter':tmp_store.squeeze()[10].cpu().detach().numpy()}], fp)
fp3.close()

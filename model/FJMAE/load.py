# import pickle
# with open('kinetics-skeleton/train/imgs/person_02_handclapping_d2_1.pkl','rb') as f:
#     data=pickle.load(f)
#     print(data)
# # import numpy as np
# data_np=np.load("kinetics-skeleton/kinetics-skeleton18/train_data.npy")
# print(data_np.shape)
from models_stmae import  mae_vit_base_patch16

import  torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
partial_weights = torch.load('/home/huaizhenhao/codes/GD/mae/output/mask0.90-xsub/ntu60_xsub.pth', map_location=device)
print(partial_weights)
# model.load_state_dict(partial_weights)

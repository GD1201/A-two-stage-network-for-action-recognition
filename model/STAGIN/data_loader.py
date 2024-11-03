# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Data loader for UCI letter, spam and MNIST datasets.
'''
# from  feeder.feeder import Feeder
# Necessary packages
import numpy as np
from util.utils import binary_sampler,binary_sampler1
# from keras.datasets import mnist
import pandas as pd

def data_loader (data_name, miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  if data_name in ['letter', 'spam','skeleton']:
    # Feeders=Feeder
    # train_info=dict()
    # train_info["data_path"]="/home/huaizhenhao/codes/GD/GAIN/Our_Data/train_data.npy"
    # train_info["label_path"]="/home/huaizhenhao/codes/GD/GAIN/Our_Data/train_label.pkl"
    # dataset_train=Feeders(**train_info)
    # data_loader_train = torch.utils.data.DataLoader(
    # dataset_train, sampler=sampler_train,
    # batch_size=batch_size,)
    # file_name = '/home/huaizhenhao/codes/GD/GAIN-master/data/pose_data_processed.csv'
    # data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    file_name='/home/huaizhenhao/codes/GD/GAIN-master/batterymodule.npy'
    data_x = np.load(file_name)  ### 200 180 18 2
    
    #   data_x= data_x[0:2000, :]
    # print("....",data_x)
    # data_x = data_x.iloc[:, 3:]
    print(data_x.shape)
  # elif data_name == 'mnist':
  #   (data_x, _), _ = mnist.load_data()
  #   data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

  # Parameters
  N,T,V,C=data_x.shape
  data_x=data_x.reshape(N,T,V*C)
  # no, dim = data_x.shape
  # Introduce missing data
  # data_m = binary_sampler(1-miss_rate, no, dim)
  # data_m = binary_sampler1(data_x, no, dim)
  data_m=binary_sampler1(data_x, N, T,V*C)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m

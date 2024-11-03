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

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import argparse
import numpy as np

from data_loader import data_loader
from gain import gain
from util.utils import rmse_loss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# gpus = tf.config.list_physical_devices('GPU')
sys.path.append('/home/huaizhenhao/codes/GD/GAIN-master')

def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  ###ori_data_x, miss_data_x, data_m 维度都为 200 18 36
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
  # print("ori_data_x:",ori_data_x.shape)
  ### 
  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters)
  # print(imputed_data_x.shape)
  # np.save('/home/huaizhenhao/codes/GD/GAIN-master/imputed_data_x.npy', imputed_data_x.reshape(imputed_data_x.shape[0],imputed_data_x.shape[1],18,2))
  imputed_data_x1=imputed_data_x.reshape(imputed_data_x.shape[0],-1)
  np.savetxt('/home/huaizhenhao/codes/GD/GAIN-master/imputed_data_x.csv', imputed_data_x1, fmt="%.2f",delimiter=',')
  # Report the RMSE performance
  rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  
  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam','skeleton'],
      default='skeleton',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.6,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=500,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse = main(args)

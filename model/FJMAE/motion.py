from util.feeder.feeder import Feeder
from feeder.feeder_ntu import Feeder1
import torch
train_info=dict()
Feeders=Feeder1
train_info["data_path"]="/mnt/dataset/private/GD/NTU60_XView.npz"
data=Feeders(**train_info)
print("数据加载成功")
# print(data.shape)
data_loader_train = torch.utils.data.DataLoader(
                dataset=Feeders(**train_info))
print(data_loader_train)
# # dataiter=iter(data_loader_train)
# # print(dataiter)
# # data,_=dataiter.next()
# # print(data.shape)
# # # print(data)
# # fdiff = data[:,:,1:,:,:] - data[:,:,:-1,:,:]
# # print(fdiff.shape)
# # fdiff = torch.abs(fdiff)
# # # print(fdiff[:,:,0:1,:,:].shape)
# # fdiff = torch.cat([fdiff[:,:,0:1,:,:], fdiff],
# # dim=2)
# # print(fdiff)
# # # fdiff = torch.linalg.vector_norm(fdiff, dim=2,
# # # keepdim=True) 
# # fdiff=torch.linalg.norm(fdiff, dim=2,
# # keepdim=True)
# # print(fdiff.shape)
# # print(data_loader_train)
# dataiter = iter(data_loader_train)
# # print(dataiter)
# data, _ = dataiter.next()
# # print(data.shape)

# # 计算相邻帧之间的差异
# fdiff = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
# print(fdiff.shape)

# # 取差异的绝对值
# fdiff = torch.abs(fdiff)
# fdiff[:,:,0:1,:,:]=0
# # 将第一个帧的差异复制一次，并与原始差异连接
# fdiff = torch.cat([fdiff[:, :, 0:1, :, :], fdiff], dim=2)
# print(fdiff.shape)
# fdiff = torch.linalg.norm(fdiff, dim=2, keepdim=True)
# print(fdiff.shape)
# sorted_indices = torch.argsort(fdiff, dim=1)

# # 使用排序后的索引对差异张量进行排序
# fdiff_sorted = torch.gather(fdiff, dim=1, index=sorted_indices)

# # 输出排序后的结果
# # print("Sorted motion weights:", fdiff_sorted)
# def motion(self,x):
#     x_motion=torch.zeros_like(x)  ## 64 30 18 256       建立全为0张量储存相应的运动信息  取绝对值 计算运动高低 由大到小排列
#     x_motion=x[:, 1:, :, :] -x[:, :-1, :, :]  #28
#     x_motion=torch.abs(x_motion) ## 绝对值  来看大小
#     # data_skeleton[:,0:1,:,:]=0
#     x_motion[:,0:1,:,:]=0
#     x_motion=torch.cat([x_motion[:,0:1,:,:],x_motion],dim=1) 

#     return x_motion

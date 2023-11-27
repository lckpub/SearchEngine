# -- coding: utf-8 --
import cv2
import numpy as np
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from PIL import Image
from matplotlib import pyplot as plt
#均衡化
def map(m):
    if 0 <= m < 0.32:
        return 0
    elif 0.32 <= m < 0.4:
        return 1
    else:
        return 2
        
#获取b g r三原色
def vector(img, h1, h2, w1, w2):
    channels = cv2.split(img[h1:h2,w1:w2])
    #b g r
    lis = []
    sum = 0
    for i in channels:
        # print(type(i))
        lis.append(i.sum())
        sum += i.sum()
    #print(lis)
    for i in range(3):
        #lis[i]/sum 为三色素占比 b g r ，利用map分成3部分区分
        lis[i] = map(lis[i]/sum)
    return lis  #最后得到的是一个列表，每个元素范围为0，1，2

def merge_vector(figname):
    img = cv2.imread(figname)
    
    h,w = img.shape[0],img.shape[1]
    mh = int(h/2) # medium height
    mw = int(w/2) # medium width
    res = []
    #分成四块,对应ppt中的四个区块
    res.extend(vector(img,0,mh,0,mw)) #H1
    res.extend(vector(img,0,mh, mw, w))#H2
    res.extend(vector(img, mh, h, 0, mw))#H3
    res.extend(vector(img, mh, h, mw, w))#H4
    return res

def LSHsearch(vector,dd):#需要返回一个bin的大小
    tmp=[0]*24    #对应的是xi,i from 1 to 24
    #d=12,c=2,d'=24
    d=12
    c=2
    for i in range(len(vector)):
        if vector[i]==0:
            tmp[2*i]=0
            tmp[2*i+1]=0
        elif vector[i]==1:
            tmp[2*i]=1
            tmp[2*i+1]=0
        else:
            tmp[2*i]=1
            tmp[2*i+1]=1

    lst=choose_subset(dd)
    hash_res_lst=[]
    # compare_lst=[x for x in lst if x!=[]]
    for i in range(d):
        if lst[i]==[]:
            continue
        else:
            for item in lst[i]:
                if (item-c*i)<=vector[i]:
                    hash_res_lst.append(1)
                else:
                    hash_res_lst.append(0)
    hashres=0
    # print(hash_res_lst)
    for i in range(len(hash_res_lst)):
        hashres+=hash_res_lst[i]*(2**(dd-1-i))
    return hashres

def choose_subset(dd):
    #Ii可以取1~12，我们只对有元素的Ii进行操作
    #选取子集的长度为几，得到的二进制数的长度就为几
    #以下为所取自己对应的各个Ii,此处选取的subset长度为6
    # lst=[[1],[3],[],[7,8],[],[],[],[],[],[],[],[]]
    if dd==4:
        lst=[[1],[3],[],[7,8],[],[],[],[],[],[],[],[]]
        # lst=[[1],[],[],[7,8],[],[],[],[],[18],[],[],[]]
        # lst=[[],[3],[],[7,8],[],[],[14],[],[],[],[],[]]
    else:
        lst=[[1],[3],[],[7,8],[],[],[14],[],[18],[],[],[]]

    return lst
    #(0,2],(2,4],(4,6],(6,8],(8,10],(10,12],(12,14],(14,16]...

#预处理
def PreProcessLSH():
    dataset=[[] for i in range(16)]#特意分成16个bin
    for i in range(0,200):
        try:
            figname='face_news/picture/{0}.png'.format(i)
            print(figname)

            #获取图片的特征向量
            vector = merge_vector(figname)
            hashres = LSHsearch(vector,4)    
            dataset[int(hashres)].append(i)
            # dataset[hashres].append(tuple([file,vector]))
        except:
            pass
    # print("PreprocessLSH done!")
    return dataset



x=PreProcessLSH()
# print(x)

y=dict()
for i in range(16):
    y[i]=x[i]
# for item in x:
#     print(item)
print(y)
np.save('hash.npy',y)
# def PreProcessNN(dir):
#     dataset = []
   
#     for i in range(1,41):
#         file='{0}/{1}.jpg'.format(dir,i)
#         vector = merge_vector(file)
#         # dataset.append(tuple([file,vector]))
#         dataset.append(i)
#     print("PreprocessNN done!")
#     return dataset

# def img_readAndShow(imgname):
#     img_bgr = cv2.imread(imgname)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.show()

# def cal_vector_angle(A,B):#计算向量之间的夹角
#     x=np.linalg.norm(A)
#     y=np.linalg.norm(B)
#     z=float(np.sum(np.multiply(A,B))/(x*y))
#     return z

# def get_array_of_fig(tmp_load):
#     len1=tmp_load.shape[1]
#     len2=tmp_load.shape[2]
#     len3=tmp_load.shape[3]
#     len=len1*len2*len3
#     # lst=[0]*len
#     lst=[]
#     for i in range(len1):
#         for j in range(len2):
#             for k in range(len3):
#                 # lst[i]=tmp_load[0][i][j][k]
#                 lst.append(tmp_load[0][i][j][k])
#     array=np.array(lst)
#     return array

# def get_compare(model):
#     compare_fig=np.load('features_model_{}_of_target.npy'.format(model))
#     compare_array=get_array_of_fig(compare_fig)
#     # compare_array=np.array(compare_fig_lst)
#     return compare_array

# def vector_method(model,aimset):
#     loadData=[]#loadData用于存储test图片中对应的归一化后的向量，每种模型对应的向量长度一致
#     for i in aimset:
#         tmp_load=np.load('model/{1}/features_model_{1}_of_{0}.npy'.format(i,model))
#         tmp_array=get_array_of_fig(tmp_load)
#         loadData.append(tmp_array)
    
#     compare_array=get_compare(model)
#     compare={}#用于储存最后得到的匹配信息
#     compare_lst=[]

#     for i in range(len(aimset)):
#         value=cal_vector_angle(loadData[i], compare_array)
#         compare[value]=aimset[i]-1
#         compare_lst.append(value)
#     compare_lst.sort(reverse=True)#求得的是cos的值，越大越相似
   
#     resnum=[]
#     for i in range(1):
#         resnum.append(compare[compare_lst[i]]+1)
#     return resnum,compare_lst

# def NN_search():# file="target.jpg"
#     t1 = time.time()
#     res,res_value=vector_method('resnet101',PreProcessNN('Dataset'))
#     for i  in range(1):
#         print("NUM{0}:img{1},vector_evaluate:{2}".format(i+1,res[i],res_value[i]))
#     print("res_picture_show:")
#     for item in res:
#         img_readAndShow('Dataset/{}.jpg'.format(item))
    
#     t2 = time.time()
#     print('time of NN:', t2 - t1)
#     print()

# def LSH_search():
#     t1 = time.time()
#     target="target.jpg"
#     vec_target=merge_vector(target)
#     hashres_target = LSHsearch(vec_target,4) 
#     # x=PreProcessLSH('Dataset')
#     # print(x)
#     aimset0=PreProcessLSH('Dataset')[int(hashres_target)]
#     # print(aimset0)

#     res0,res_value0=vector_method('resnet101',aimset0)
#     for i  in range(1):
#         print("NUM{0}:img{1},vector_evaluate:{2}".format(i+1,res0[i],res_value0[i]))
#     print("res_picture_show:")
#     for item in res0:
#         img_readAndShow('Dataset/{}.jpg'.format(item))
    
#     t2 = time.time()
#     print('time of LSH:', t2 - t1)

# def main():
#     # NN_search()
#     LSH_search()

# if __name__=="__main__":
#     main()
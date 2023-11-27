# SJTU EE208

import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from PIL import Image

# Method=['resnet50','vgg16','alexnet','resnet101','squeezenet1_1','googlenet','mnasnet1_0','densenet121']
# Method=['googlenet','mnasnet1_0','squeezenet1_1','densenet121']
Method=['vgg16']
def features(x,method):
    if method=='vgg16' :
        x = model.features(x)
        x = model.avgpool(x)
    else:
        x=x
    
    return x
        

for method in Method:
    print('Load model:{}'.format(method))
    # model = torch.hub.load('pytorch/vision', method, pretrained=True)
    model = torchvision.models.vgg16(pretrained=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    print('Prepare image data!')
    # #获得测试集中的feature
    for i in range(0,200):
        figname='face_news/picture/{0}.png'.format(i)
        test_image = default_loader(figname)
        input_image = trans(test_image)
        input_image = torch.unsqueeze(input_image, 0)

        # print('Extract features!')
        # start = time.time()
        image_feature = features(input_image,method)
        image_feature = image_feature.detach().numpy()
        # print('Time for extracting features: {:.2f}'.format(time.time() - start))


        print('Save features!')
        np.save('face_news/picture features/feature_of_img{0}.npy'.format(i), image_feature)

        # loadData=np.load('features.npy')
        # print(loadData)
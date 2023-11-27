import os
import requests
# path=os.getcwd()+'\\picture\\'   #设置图片文件路径，前提是必须要有abc这个文件夹
path="face_news/picture/"
import pandas as pd
df=pd.read_excel('face_news/url1.xlsx')

urls=df['url']
for i in range(len(urls)):   
    try:
        r = requests.request('get',urls[i])  #获取网页
        print(i)
        print(r.status_code)
        with open(path+str(i)+'.png','wb') as f:  #打开写入到path路径里-二进制文件，返回的句柄名为f
            f.write(r.content)  #往f里写入r对象的二进制文件
    except:
        pass
    # f.close()
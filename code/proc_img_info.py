import pymysql
import numpy as np
db = pymysql.connect(host='152.136.97.17',
                         port=3306,
                         user='cjx',
                         password='111111',
                         db='crawler_pages',)

cursor = db.cursor()
saver=dict()
sql="SELECT * FROM pageInfo;"
cursor.execute(sql)
print("start")
data_page = cursor.fetchall()
sql="SELECT * FROM imgInfo;"
cursor.execute(sql)
print("fetch1")
data_img = cursor.fetchall()
print("fetch2")
def backtrace(news_id):
    sql = f"SELECT * FROM pageInfo WHERE id = {news_id};"  # 使用WHERE可以加限制条件
    cursor.execute(sql)
    data = cursor.fetchone()
    title=data[1]
    url=data[2]
    date=data[4]
    return title,url,date

for img in data_img:
    number=img[2]
    print("processing",number)
    news_id=img[0]
    imgurl=img[1]
    trace=list(backtrace(news_id))
    info=[news_id,imgurl,trace[0],trace[1],trace[2]]
    saver[number]=info
np.save("process.npy",saver)
# SJTU EE208

import sys, os, threading, time,re
import lucene
from datetime import datetime
import pymysql

# from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.document import Document, Field, FieldType, StringField, TextField
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
import urllib
# from tld import get_tld
import jieba


def JiebaAnalyzer(contents):
    seglist=jieba.cut_for_search(contents)
    return ' '.join(seglist)

"""
This class is loosely based on the Lucene (java implementation) demo class 
org.apache.lucene.demo.IndexFiles.  It will take a directory as an argument
and will index all of the files in that directory and downward recursively.
It will index on the file path, the file name and the file contents.  The
resulting Lucene index will be placed in the current directory and called
'index'.
"""

class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)

class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, storeDir):

        if not os.path.exists(storeDir):
            os.mkdir(storeDir)
    

        # store = SimpleFSDirectory(File(storeDir).toPath())
        store = SimpleFSDirectory(Paths.get(storeDir))
        analyzer = WhitespaceAnalyzer()
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)
        self.indexDocs(writer)
        ticker = Ticker()
        print('commit index')
        threading.Thread(target=ticker.run).start()
        writer.commit()
        writer.close()
        ticker.tick = False
        print('done')

    # def findUrlAndTitle(self, news_id):
    #     url =''
    #     title = ''
    #     for content in self.contents:
    #         if news_id in content:
    #             end = start = 0
    #             try:
    #                 while(len(content)!=None and content[start]!='\t'):
    #                     start += 1
    #                 url = content[2:start-1]
    #                 end = start = start + len('\t')
    #             except:
    #                 pass
    #             try:
    #                 while(content[end] != '\t'):
    #                     end+=1
    #                 title = content[start:end]
    #             except:
    #                 title = None
    #     return url,title
    
    def findDomainByUrllib(self,url):
        return urllib.parse.urlparse(url).netloc
    
    # 找数据库中的新闻信息 #
    def findNewsInfo(self):
        # 创建数据库的连接 #
        db = pymysql.connect(host='152.136.97.17',
                             port=3306,
                             user='cjx',
                             password='111111',
                             db='crawler_pages',)

        cursor = db.cursor()

        #查看所有列的信息的命令
        sql = "SELECT * FROM pageInfo;" # limit 调整获取的新闻数量
        #执行mysql命令
        cursor.execute(sql)
        #得到结果
        data = cursor.fetchall()
        #返回的是一个数据的集合
        return data

    def indexDocs(self, writer):

        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.NONE)  # Not Indexed
        t2 = FieldType()
        t2.setStored(True)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)  # Indexes documents, frequencies and positions.
        # for root, dirnames, news_ids in os.walk(root):
            # for filename in filenames:
            #     if not filename.endswith('.txt'):
            #         continue
                # 这里要找每篇新闻的id...... #
        data = self.findNewsInfo() 
        for item in data:
            news_id = item[0]
            title = item[1]
            url = item[2]
            contents = item[3]
            date=item[4]
            print("adding", news_id)

            try: # 处理每一篇新闻 #
                contents=JiebaAnalyzer(contents)
                # print(url)
                domain = self.findDomainByUrllib(url)
                doc = Document()
                doc.add(Field("news_id", news_id, t1))
                doc.add(Field("url",url,t1))
                doc.add(Field("title",title,t1))
                doc.add(TextField("site",domain,Field.Store.YES))
                if len(contents) > 0:
                    doc.add(Field("contents", contents, t2))
                else:
                    print("warning: no content in %s" % news_id)
                doc.add(Field("date",date,t1))
                writer.addDocument(doc)
            except Exception as e:
                print("Failed in indexDocs:", e)



if __name__ == '__main__':
    lucene.initVM()#vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    # import ipdb
    # ipdb.set_trace()
    start = datetime.now()
    try:
        IndexFiles( "index")
        end = datetime.now()
        print(end - start)
    except Exception as e:
        print("Failed: ", e)
        raise e

# SJTU EE208

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time
from datetime import datetime
# from java.io import File
import jieba
from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer,SimpleAnalyzer
from org.apache.lucene.document import Document, Field, FieldType, StringField, TextField
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
from bs4 import BeautifulSoup
import re
import urllib.error
import urllib.parse
import urllib.request
import pymysql
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

    def __init__(self,  storeDir, analyzer):

        if not os.path.exists(storeDir):
            os.mkdir(storeDir)

        store = SimpleFSDirectory(File(storeDir).toPath())
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
    def findDomainByUrllib(self,url):
        return urllib.parse.urlparse(url).netloc
    def findimgInfo(self):
        # 创建数据库的连接 #
        db = pymysql.connect(host='152.136.97.17',
                             port=3306,
                             user='cjx',
                             password='111111',
                             db='crawler_pages',)

        cursor = db.cursor()

        #查看所有列的信息的命令
        sql = "SELECT * FROM imgInfo;" # limit 调整获取的新闻数量
        #执行mysql命令
        cursor.execute(sql)
        #得到结果
        data = cursor.fetchall()
        #返回的是一个数据的集合
        return data
    def backtrace(self,news_id):
        db = pymysql.connect(host='152.136.97.17',
                             port=3306,
                             user='cjx',
                             password='111111',
                             db='crawler_pages',)

        cursor = db.cursor()
        sql = f"SELECT * FROM pageInfo WHERE id = {news_id};"  # 使用WHERE可以加限制条件
        cursor.execute(sql)
        data = cursor.fetchone()
        title=data[1]
        url=data[2]
        contents=data[3]
        date=data[4]
        return title,url,contents,date
    def getTxtAttribute(self, contents, attr):
        m = re.search(attr + ': (.*?)\n',contents)
        if m:
            return m.group(1)
        else:
            return ''
    def indexDocs(self, writer):
        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.NONE)  # Not Indexed   
        t2 = FieldType()
        t2.setStored(True)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        data = self.findimgInfo()
        for item in data:
            news_id = item[0]
            imgurl= item[1]
            try: 
                trace_data=self.backtrace(news_id)
                title=trace_data[0]
                url=trace_data[1]
                contents=trace_data[2]
                contents = JiebaAnalyzer(contents)
                date=trace_data[3]
                print("adding", imgurl)
            
                doc = Document()
                domain = self.findDomainByUrllib(url)
                doc.add(Field("news_id", news_id, t1))
                doc.add(Field("imgurl",imgurl,t1))
                doc.add(Field("title",title,t1))
                doc.add(Field("url",url,t1))
                doc.add(TextField("site",domain,Field.Store.YES))
                if len(contents) > 0:
                    doc.add(Field("contents", contents, t2))
                else:
                    contents=" "
                    doc.add(Field("contents", contents, t2))
                    print("warning: no content in %s" % news_id)
                doc.add(Field("date",date,t1))
                writer.addDocument(doc)
            except Exception as e:
                print("Failed in indexDocs:", e)
      
                

if __name__ == '__main__':
    """
    if len(sys.argv) < 2:
        print IndexFiles.__doc__
        sys.exit(1)
    """
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    start = datetime.now()
    try:
        """
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        IndexFiles(sys.argv[1], os.path.join(base_dir, INDEX_DIR),
                   StandardAnalyzer(Version.LUCENE_CURRENT))
                   """
        analyzer = SimpleAnalyzer()
        IndexFiles("index_img", analyzer)
        end = datetime.now()
        print(end - start)
    except Exception as e:
        print("Failed: ", e)
        raise e
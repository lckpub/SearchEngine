# SJTU EE208

import sys, os, lucene

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.search import BooleanClause
from org.apache.lucene.search.highlight import Highlighter, QueryScorer, SimpleFragmenter, SimpleHTMLFormatter
vm = lucene.initVM(vmargs=['-Djava.awt.headless=true'])

import jieba
import pymysql

def JiebaAnalyzer(contents):
    seglist=jieba.cut_for_search(contents)
    return ' '.join(seglist)

"""
This script is loosely based on the Lucene (java implementation) demo class 
org.apache.lucene.demo.SearchFiles.  It will prompt for a search query, then it
will search the Lucene index in the current directory called 'index' for the
search query entered against the 'contents' field.  It will then display the
'path' and 'name' fields for each of the hits it finds in the index.  Note that
search.close() is currently commented out because it causes a stack overflow in
some cases.
"""

def parseCommand(command):
    command_dic = {}

    for i in command.split(' '):
        if ':' in i:
            opt,value = i.split(':')[0:2]
            if(opt == "site") and value!='':
                command_dic[opt] = value
        else:
            command_dic['contents'] = command_dic.get('contents','') + ' ' + i
    return command_dic
def getNewsContent(url):
        # 创建数据库的连接 #
    db = pymysql.connect(host='152.136.97.17',
                            port=3306,
                            user='cjx',
                            password='111111',
                            db='crawler_pages',)

    cursor = db.cursor()

    #查看所有列的信息的命令
    sql = f"SELECT content FROM pageInfo WHERE url='{url}';" # limit 获取的对应的新闻内容
    #执行mysql命令
    cursor.execute(sql)
    #得到结果
    data = cursor.fetchone()
    return data[0]



def run_text(searcher, analyzer, command):
    if not command:
        return
    print('command:',command)
    command_dict = parseCommand(command)
    print('commanddict',command_dict)
    querys = BooleanQuery.Builder()
    for k,v in command_dict.items():
        if(k == 'contents'):
            v_list = JiebaAnalyzer(v)
            query = QueryParser(k, analyzer).parse(v_list)
            querys.add(query, BooleanClause.Occur.MUST)
        else:
            query = QueryParser(k, analyzer).parse(v)
            querys.add(query, BooleanClause.Occur.MUST)

    scoreDocs = searcher.search(querys.build(), 50).scoreDocs
    formatter =SimpleHTMLFormatter("<font color = 'red'><em>", "</em></font>")
    highlighter = Highlighter(formatter, QueryScorer(querys.build()))
    highlighter.setTextFragmenter(SimpleFragmenter(50))
    result = []
    for i,scoreDoc in enumerate(scoreDocs):
        doc = searcher.doc(scoreDoc.doc)
        data = {}
        data['title'] = doc.get("title")
        data['url'] = doc.get('url')
        contents=str(doc.get("contents"))
        stream=analyzer.tokenStream("contents", contents)
        related=highlighter.getBestFragment(stream, contents)
        #data['related_text'] = related_text(doc.get('url'),command)
        data["related_text"]=str(related)
        data['date'] = doc.get('date')
        result.append(data)
    return result
            # print 'explain:', searcher.explain(query, scoreDoc.doc)
   
def run_img(searcher,analyzer,command):
    if command == '':
        return
    print()
    print ("Searching for:", command)
    command_dict = parseCommand(command)
    querys = BooleanQuery.Builder()
    for k,v in command_dict.items():
        if(k == 'contents'):
            v_list = JiebaAnalyzer(v)
            query = QueryParser(k, analyzer).parse(v_list)
            querys.add(query, BooleanClause.Occur.MUST)
        else:
            query = QueryParser(k, analyzer).parse(v)
            querys.add(query, BooleanClause.Occur.MUST)
    scoreDocs = searcher.search(querys.build(), 50).scoreDocs
    print ("%s total matching documents." % len(scoreDocs))
    results=[]
    for i,scoreDoc in enumerate(scoreDocs):
        doc = searcher.doc(scoreDoc.doc)
        url=str(doc.get("url"))
        imgurl=str(doc.get("imgurl"))
        title=str(doc.get("title"))
        score=scoreDoc.score
        date=str(doc.get("date"))
        val=dict()
        val["imgurl"]=imgurl
        val["title"]=title
        val["url"]=url
        val["score"]=score
        val["date"]=date
        results.append(val)
    return results

class Searcher:
    
    def __init__(self,store_dir):
        print ('lucene', lucene.VERSION)
        vm.attachCurrentThread()
        #base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.directory = SimpleFSDirectory(File(store_dir).toPath())
        self.searcher = IndexSearcher(DirectoryReader.open(self.directory))
        self.analyzer =  WhitespaceAnalyzer()#Version.LUCENE_CURRENT)
    
    def SearchQueryText(self,command):
        return run_text(self.searcher,self.analyzer,command)
    def SearchQueryImg(self,command):
        return run_img(self.searcher,self.analyzer,command)
    def __del__(self):
        del self.searcher

# if __name__ == '__main__':
#     STORE_DIR = "index"
#     lucene.initVM(vmargs=['-Djava.awt.headless=true'])
#     print ('lucene', lucene.VERSION)
#     #base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
#     directory = SimpleFSDirectory(File(STORE_DIR).toPath())
#     searcher = IndexSearcher(DirectoryReader.open(directory))
#     analyzer =  WhitespaceAnalyzer()#Version.LUCENE_CURRENT)
#     run(searcher, analyzer)
#     del searcher
    

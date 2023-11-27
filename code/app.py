# SJTU EE208

from flask import Flask, redirect, render_template, request, url_for
import SearchFiles
import picture_to_picture

def page_date_key(result):
    return result["date"]
app = Flask(__name__)

# @app.before_first_request
# def load_index_form():
#     global Searcher
#     Searcher=SearchFiles.Searcher()

@app.route('/', methods=['POST', 'GET'])
def index_form():
    if request.method == "POST":
        Query = request.form['Query']
        
        return redirect(url_for('showres', Query = Query))
    return render_template("search_page.html")

@app.route('/match', methods=['POST', 'GET'])
def match():
    if request.method == "POST":
        Query = request.form['Query']
        
        return redirect(url_for('show_match', Query = Query))
    return render_template("match.html")

@app.route('/showres_text', methods=['GET'])
def showres_text():
    # vm.attachCurrentThread()
    Searcher=SearchFiles.Searcher('index')
    Query = request.args.get('Query')
    dataSet = Searcher.SearchQueryText(Query)
    return render_template("show_res.html", Query = Query, dataSet = dataSet)

@app.route('/showres_text_datesorted', methods=['GET'])
def showres_text_datesorted():
    # vm.attachCurrentThread()
    Searcher=SearchFiles.Searcher('index')
    Query = request.args.get('Query')
    dataSet = Searcher.SearchQueryText(Query)
    dataSet=sorted(dataSet,key=page_date_key,reverse=True)
    print("这里有没有东西",dataSet)
    return render_template("show_res_datesorted.html", Query = Query, dataSet = dataSet)

@app.route('/showres_img', methods=['GET'])
def showres_img():
    # vm.attachCurrentThread()
    Searcher=SearchFiles.Searcher('index_img')
    Query = request.args.get('Query')
    dataSet = Searcher.SearchQueryImg(Query)
    print(dataSet)
    return render_template("resultim.html", Query = Query, results = dataSet)

@app.route('/showres_img_datesorted', methods=['GET'])
def showres_img_datesorted():
    # vm.attachCurrentThread()
    Searcher=SearchFiles.Searcher('index_img')
    Query = request.args.get('Query')
    dataSet = Searcher.SearchQueryImg(Query)
    dataSet=sorted(dataSet,key=page_date_key,reverse=True)
    print(dataSet)
    return render_template("resultim_datesorted.html", Query = Query, results = dataSet)

@app.route('/show_match', methods=['GET'])
def show_match():
    # vm.attachCurrentThread()
    Query = request.args.get('Query')
    dataSet=picture_to_picture(str(Query))
    return render_template("show_match.html", Query = Query, results = dataSet)

@app.route('/show_match_datesorted', methods=['GET'])
def show_match_datesorted():
    # vm.attachCurrentThread()
    Query = request.args.get('Query')
    dataSet = picture_to_picture(str(Query))
    dataSet=sorted(dataSet,key=page_date_key,reverse=True)
    return render_template("show_match_datesorted.html", Query = Query, results = dataSet)
if __name__ == '__main__':
    app.run(debug=True,port=6699)

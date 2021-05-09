from flask import Flask , request, jsonify
import util

app =  Flask(__name__)

@app.route('/classify_image',methods = ['GET','POST'])
def classfy_image():
    image_data = request.form['image_data']
    responce = jsonify(util.classify_image(image_data))
    responce.headers.add('Access-Control-Allow-Origin','*')
    return responce


if __name__=="__main__":
    print("Starting Flask Server for Image detection ...... !!!!!")
    util.load_artifacts()
    app.run(port=5000)

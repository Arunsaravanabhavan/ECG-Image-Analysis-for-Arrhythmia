import os
import numpy as np
from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app=Flask(__name__)
model=load_model('ECG.h5')

@app.route("/")
def about():
    return render_template("about.html")

@app.route("/about")
def home():
    return render_template("about.html")

@app.route("/info")
def information():
    return render_template("info.html")

@app.route("/upload")
def test():
    return render_template("upload.html")

@app.route("/predict",methods=["GET","POST"])
@app.route("/predict",methods=["GET","POST"])
def upload():
    if request.method=='POST':
        f=request.files['file']
        basepath=os.path.dirname('_file_')
        filepath=os.path.join(basepath,"uploads",f.filename)
        f.save(filepath)
    
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
    
        preds=model.predict(x)
        pred=np.argmax(preds,axis=1)
        print("prediction",pred)
    
        index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction',
               'Premature Ventricular Contractions', 'Right Bundle Branch Block','Ventricular Fibrillation']
        result=str(index[pred[0]])
        return result
    return None

if __name__=="__main__":
    app.run(debug=False)
            
            
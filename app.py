from flask import Flask,render_template,request
import os
import numpy as np
import pandas as pd
from CustomerChurnPrediction.pipeline.prediction import PredictionPipeline



app=Flask(__name__)

@app.route('/',methods=['GET']) #route to display the home page
def homepage():
    return render_template('index.html')



if __name__=='__main__':
    #app.run(host='0.0.0.0',port=5000,debug=True)
    app.run(host='0.0.0.0',port=5000)


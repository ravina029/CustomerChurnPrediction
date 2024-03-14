from flask import Flask,render_template,request
import os
import numpy as np
import pandas as pd
from CustomerChurnPrediction.pipeline.prediction import PredictionPipeline



app=Flask(__name__)

@app.route('/',methods=['GET']) #route to display the home page
def homepage():
    return render_template('index.html')


@app.route('/train', methods=['GET'])  #route to train the pipeline
def training():
    os.system('python main.py')
    return "Training Successful"


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Age =float(request.form['Age'])
            Balance =float(request.form['Balance'])
            NumOfProducts =float(request.form['NumOfProducts'])
            IsActiveMember =float(request.form['IsActiveMember'])
            Complain =float(request.form['Complain'])
            
         
            data = [Age, Balance, NumOfProducts, IsActiveMember, Complain]
            data = np.array(data).reshape(1, 5)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__=='__main__':
    #app.run(host='0.0.0.0',port=5000,debug=True)
    app.run(host='0.0.0.0',port=5000)



#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import sklearn
import joblib

from flask import Flask,request,render_template

app = Flask(__name__)


# In[34]:


@app.route('/')
def index():
    return render_template('index.html')


# @app.route("/")
# def index():
#     return render_template("index.html")

# In[31]:


@app.route('/prediccion',methods=['GET','POST'])
def predict():
    if request.method =='POST':
        try:
            var_1=float(request.form['var_1'])
            var_2=float(request.form['var_2'])
             
                
            pred_args=[var_1,var_2]
            pred_arr=np.array(pred_args)
            preds=pred_arr.reshape(1,-1)
            modelo=open("./modelo.pkl","rb")
            modelo_clas=joblib.load(modelo)
            prediccion_modelo=modelo_clas.predict(preds)
            prediccion_modelo=round(float(prediccion_modelo),2)
            if prediccion_modelo == 1.0:
                prediccion_modelo = "Aprueba"
            else:
                prediccion_modelo = "No Aprueba"
        except ValueError:
            return "por favor entra nombre validos"
        return render_template("prediccion.html",prediccion=prediccion_modelo)


# In[32]:


if __name__=='__main__':
    app.run(debug=True)


# In[ ]:





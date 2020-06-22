from flask import Flask, request, jsonify, render_template
import pickle
from gevent.pywsgi import WSGIServer
from sklearn.model_selection import train_test_split
import numpy as np,pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
     #For rendering results on HTML GUI
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    # instantiate labelencoder object
    lf1 = LabelEncoder()
    lf2 = LabelEncoder()
    #lf3 = LabelEncoder()
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    lf1.classes_=np.load('classes1.npy')
    lf2.classes_=np.load('classes2.npy')
    #lf3.classes_=np.load('classes3.npy')
    np.load = np_load_old
            
    Nom = request.form ['Nom du Port']
    Navire = request.form ['Nom du Navire']
    day = request.form ['day']
    month = request.form ['month']
    year = request.form ['year']
    l=[[Nom,Navire,day,month,year]]
    n=pd.DataFrame(data=l, columns=['Nom','Nom du Navire','day','month','year'])
    n['Nom' ] = lf1.transform(n['Nom' ])
    n['Nom du Navire'] = lf2.transform(n['Nom du Navire'])
    #n['ETA'] = lf3.transform(n['ETA'])

    categorical_feature_mask = n.dtypes==object
    categorical_cols =n.columns[categorical_feature_mask].tolist()
    #n[categorical_cols] = n[categorical_cols].apply(lambda col: le.transform(col))
    y_pred = model.predict(n)

    return render_template('index.html', prediction_text='Import :  {}'.format(y_pred))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)



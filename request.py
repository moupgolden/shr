import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Nom de port':1,'Nom du Navire':2, 'ETA':20/11/2020})

print(r.json())


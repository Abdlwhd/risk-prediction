from flask import Flask,request,jsonify
import numpy as np
import pickle
from flask_restful import Resource, Api
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
api = Api(app)
class RiskAPI(Resource):
# @app.route('/predict',methods=['POST'])
# def predict():
        def post(self):
            AGE = request.form.get('AGE')
    #     PTGENDER = request.form.get('PTGENDER')
            PTEDUCAT = request.form.get('PTEDUCAT')
            PTRACCAT = request.form.get('PTRACCAT')
            PTMARRY = request.form.get('PTMARRY')
            TRABSCOR = request.form.get('TRABSCOR')
            FAQ = request.form.get('FAQ')
            MOCA = request.form.get('MOCA')
            inputs =  np.array([AGE,PTEDUCAT,PTRACCAT,PTMARRY,TRABSCOR,FAQ,MOCA])
            result = model.predict_proba(inputs.reshape(1, -1))[:,1]
            return jsonify({'Risk of AD':str(result[0]*100)})
api.add_resource(RiskAPI, '/predict')
if __name__ == '__main__':
     app.run()
#     from waitress import serve
#     serve(app, host="0.0.0.0", port=8080)
from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

from src.logger import logging

application = Flask(__name__)

app= application

@app.route('/')

def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    

    else: 
        data = CustomData(
            EDUCATION=str(request.form.get('EDUCATION')),
            MARRIAGE=str(request.form.get('MARRIAGE')),
            BILL_AMT2=float(request.form.get('BILL_AMT2')),
            BILL_AMT4=float(request.form.get('BILL_AMT4')),
            BILL_AMT5=float(request.form.get('BILL_AMT5')),
            BILL_AMT6=float(request.form.get('BILL_AMT6')),
            PAY_0=float(request.form.get('PAY_0')),
            PAY_2=float(request.form.get('PAY_2')),
            PAY_3=float(request.form.get('PAY_3')),
            PAY_4=float(request.form.get('PAY_4')),
            PAY_5=float(request.form.get('PAY_5')),
            PAY_6=float(request.form.get('PAY_6'))
        )
        final_new_data = data.get_data_as_dataframe()

        logging.info(
            f"final new data - {type(final_new_data)} and value-{final_new_data}")

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('results.html', final_result=results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
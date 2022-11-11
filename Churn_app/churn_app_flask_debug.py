import pandas as pd
import pickle
import os
import jsonpickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from flask import Flask, flash, render_template, request, send_file, session, redirect, jsonify
from werkzeug.utils import secure_filename

# app config
app = Flask('Churn')
app.config.from_pyfile('app_config.py')

@app.route('/ui', methods=['GET', 'POST'])
def user_input():
    """ Obtain user input """

    return render_template('user_input.html')

@app.route('/prs', methods=['GET', 'POST'])
def parse():
    """ Parse user input then pass to predict function """
    
    # deny direct user access to parse url
    if request.referrer != ''.join(request.host_url + 'ui'):
        flash('Error! Please fill all form fields first.')
        return redirect('/ui')

    else:

        # Classification threshold
        threshold = request.form['threshold']

        # simple checks on user input
        if request.form['threshold'] == '':
            flash('Please specify classification threshold.')
            return redirect(request.referrer)
        
        if (float(threshold) < 0) or (float(threshold) > 1):
            flash('wrong input, threshold must be between 0 and 1 (i.e.: 0.5)')
            return redirect(request.referrer)

        # customers file (feature matrix)
        customers = request.files['customers_json']

        filename = secure_filename(customers.filename)

        # simple checks on uploaded file
        file_ext = os.path.splitext(filename)[1]

        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            flash('Missing/Wrong upload! Please select \'.JSON\' file.')
            return redirect(request.referrer)

        # save uploaded customers files
        customers.save(filename)

        # load model
        model_path = ''.join(os.path.dirname(os.path.abspath(__file__)) + '\\' + 'churn_model.bin')

        with open(model_path, 'rb') as f_in:
            dv, model = pickle.load(f_in)

        # load customers data
        cust_path = ''.join(app.config['UPLOAD_FOLDER'] + '\\' + filename)

        with open(cust_path, 'rb') as file:
            f_read = file.read()
            customers = json.loads(f_read)

        # JSON serialization of sklearn modules and user input
        session['dv'] = jsonpickle.encode(dv)
        session['model'] = jsonpickle.encode(model)
        session['threshold'] = threshold
        session['customers'] = customers
        
        return redirect('/prd')
        
@app.route('/prd', methods=['GET', 'POST'])
def predict():
    """ Generate predictions and report back to the user """

    # deny direct user access to predict url
    if request.referrer != ''.join(request.host_url + 'ui'):
        flash('Error! Please fill all form fields first.')
        return redirect('/ui')

    else:

        # decode customer input
        dv = DictVectorizer()
        model = LogisticRegression()
        
        dv = jsonpickle.decode(session.get('dv'))
        model = jsonpickle.decode(session.get('model'))
        threshold = session.get('threshold')
        customers = session.get('customers')

        # generate predictions
        results = {}

        for cust in customers.items():

            x = dv.transform(cust[1])
            p_prob = float(model.predict_proba(x)[:, 1][0])

            # 'bool' to ensure that result is properly converted to text
            churn = bool(p_prob >= float(threshold))

            # predict
            results.update({f'{cust[0]}': {f'churn proba': round(p_prob, 2), f'chrun': churn}})

        # return results as a dataframe
        res_df = pd.DataFrame.from_dict(results,'index').rename_axis('customer').reset_index()
        res_df.to_csv('res_df.csv', index = False)

        return send_file(''.join(os.path.dirname(os.path.abspath(__file__)) + '\\res_df.csv'  ))

        # # display results directly
        # return jsonify(results)

if __name__ == '__main__':
        app.run(debug = True, host = '0.0.0.0', port = 8888, use_reloader=False)

# for local deployment on Windows OS use 'waitress-serve --listen=*:8888 churn_app_flask_debug:app' in cmd from app folder
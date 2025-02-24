#!/usr/bin/env python
from flask import Flask, request, redirect, url_for, render_template, send_from_directory,send_file,flash,session
from flask_session import Session  # https://pythonhosted.org/Flask-Session
from werkzeug.utils import secure_filename
import requests
import app_config
import os
#%load_ext autoreload
#%autoreload 2
import auxiliar_functions
from auxiliar_functions import *
import data_import
import process_features_transformation
import process_model_performance_analysis
import process_run_new_entries

application = Flask(__name__)
application.config.from_object(app_config)
Session(application)

application.config['ALLOWED_EXTENSIONS'] = set(['xlsx','xls','csv'])



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in application.config['ALLOWED_EXTENSIONS']


@application.route("/" , methods=['GET', 'POST'])
def  study_overview():
    #data_import.upload_file()
    return render_template('case_study_intro.html')

@application.route("/raw_data" , methods=['GET', 'POST'])
def raw_data():
    data_import.upload_file()
    return render_template('plan_analysis.html')

@application.route("/features_transforming" , methods=['GET', 'POST'])
def running_transformations():
    process_features_transformation.run_data_transformations()
    #process_features_transformation.features_selection()
    #process_features_transformation.model()
    return render_template('features_transforming.html')


@application.route("/model_performance_analysis" , methods=['GET', 'POST'])
def  running_plots():
    process_model_performance_analysis.model_performance_analysis()
    return render_template('model_performance_analysis.html')


#@application.route("/new_entries", methods=['GET', 'POST'])
#def new_entry():
#    result = None
#    if request.method == 'POST':
#        additional_data = request.form.get('additional_data', '')
#        if additional_data:
#            data_to_store = {'skills': additional_data}
#
#            with open('./data/newEntries.pkl', 'wb') as f:
#                pickle.dump(data_to_store, f)
#
#            flash('Data stored and prediction made successfully!')
#            
#            # Run the prediction and get the result
#            result = process_run_new_entries.new_skills()
#        else:
#            flash('Please enter skills data.')
#    
#    return render_template('model_prediction_new_entry.html', result=result)

@application.route("/new_entries", methods=['GET', 'POST'])
def new_entry():
    result = None
    if request.method == 'POST':
        submit_type = request.form.get('submit_type')
        
        if submit_type == 'file':
            if 'file' in request.files:
                file = request.files['file']
                if file and file.filename.endswith('.csv'):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join('./data', filename)
                    file.save(file_path)
                    
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    data_to_store = {'skills': df['HaveWorkedWith'].tolist()}
                    
                    with open('./data/newEntries.pkl', 'wb') as f:
                        pickle.dump(data_to_store, f)
                    
                    flash('File uploaded and data stored successfully!')
                    result = process_run_new_entries.new_skills()
                else:
                    flash('Please upload a valid CSV file.')
            else:
                flash('No file uploaded.')
        
        elif submit_type == 'single':
            additional_data = request.form.get('additional_data', '')
            if additional_data:
                data_to_store = {'skills': additional_data}
                
                with open('./data/newEntries.pkl', 'wb') as f:
                    pickle.dump(data_to_store, f)
                
                flash('Data stored and prediction made successfully!')
                result = process_run_new_entries.new_skills()
            else:
                flash('Please enter skills data.')
        
        else:
            flash('Invalid submission type.')
    
    return render_template('model_prediction_new_entry.html', result=result)

import random, threading, webbrowser

port = 5000 + random.randint(0, 999)
url = "http://127.0.0.1:{0}".format(port)

threading.Timer(1.25, lambda: webbrowser.open(url) ).start()

if __name__=='__main__':
    #application.run(debug=True,threaded = True)
	application.run(port=port, debug=False)
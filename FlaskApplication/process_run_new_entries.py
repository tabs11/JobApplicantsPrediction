#%load_ext autoreload
#%autoreload 2
import auxiliar_functions
from auxiliar_functions import *

def new_skills():

    with open('./Transformations/transformer_and_features.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    transformer = loaded_dict['TFIDFtransformer']
    feature_names = loaded_dict['TFIDFfeature_names']

    with open('./Transformations/model_selected_features.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    model_selected_features = loaded_dict['model_selected_features']
    results = loaded_dict['performance_results']


    with open('./Transformations/model_tuning.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    tuned_model     = loaded_dict['tuned_model']

    with open('./data/newEntries.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    Xentry     = loaded_dict['skills']
    #Xentry = Xentry.strip("'\"")
    Xentry = Xentry.lstrip("'\"").rstrip("'\"")
    Xentry = Xentry.lstrip('"\"').rstrip('"\"')
    if type(Xentry)==str:
        Xentry = pd.DataFrame({'HaveWorkedWith': [Xentry]})

    XentryTFIDF = transformer.transform(Xentry)
    XentryTFIDF = pd.DataFrame(XentryTFIDF.toarray(), columns=feature_names, index=Xentry.index)
    XentryTFIDF= XentryTFIDF[model_selected_features['Decision Tree']]
    prediction = tuned_model.predict(XentryTFIDF)
    prediction = pd.DataFrame(prediction, columns=['Employed'])
    result = pd.concat([Xentry.reset_index(), prediction], axis=1)
    result=result.reset_index(drop=True)
    result = result.drop('index',axis=1)

    fig = create_table(result,'Employement Prediction',
        width=900, 
        height=400,
        columnwidth=[30,4])


    return fig.to_html(full_html=False)
    #fig.write_html("./templates/model_prediction_new_entry.html")
    ##    
    #import bs4
    #with open('./templates/model_prediction_new_entry.html',"r",encoding='utf-8') as inf:
    #    txt = inf.read()
    #    soup = bs4.BeautifulSoup(txt,features="lxml")
    #toreplace= '{% extends "login_layout.html" %}\n{% block content %}\n<p>&nbsp;</p>\n<div class="multi_upload_index">\n<h2 style="color:black;background-color:#cccccc" class="text-muted"><b><font size="5">Prediction:</font></b></h2>\n{% with messages = get_flashed_messages() %}\n{% if messages %}\n{% for message in messages %}\n<div class="alert alert-info">{{ message }}</div>\n{% endfor %}\n{% endif %}\n{% endwith %}\n<form method="POST" action="{{ url_for("new_entry") }}">\n<div class="form-group">\n<label for="store_data">Store Data:</label>\n<input type="checkbox" id="store_data" name="store_data">\n</div> \n<div class="form-group">\n<label for="additional_data">Additional Data (Skills):</label>\n<textarea id="additional_data" name="additional_data" class="form-control" rows="4"></textarea>\n</div>\n<button type="submit" class="btn btn-primary">Submit</button>\n</form>\n<div id="prediction-results">\n</div>\n</div>'
    #toreplace= '{% extends "login_layout.html" %}\n{% block content %}\n<p>&nbsp;</p>\n<div class="multi_upload_index">\n<h2 style="color:black;background-color:#cccccc" class="text-muted"><b><font size="5">Prediction:</font></b></h2>\n{% with messages = get_flashed_messages() %}\n{% if messages %}\n{% for message in messages %}\n<div class="alert alert-info">{{ message }}</div>\n{% endfor %}\n{% endif %}\n{% endwith %}\n<form method="POST" action="{{ url_for("new_entry") }}">\n<div class="form-group">\n<label for="store_data">Store Data:</label>\n<input type="checkbox" id="store_data" name="store_data">\n</div> \n<div class="form-group">\n<label for="additional_data">Additional Data (Skills):</label>\n<textarea id="additional_data" name="additional_data" class="form-control" rows="4"></textarea>\n</div>\n<button type="submit" class="btn btn-primary">Submit</button>\n</form>\n<div id="prediction-results">\n</div>\n</div>'
#
    #example=str(soup).replace('<html>\n<head><meta charset="utf-8"/></head>\n',toreplace).replace('</html>','{% endblock %}')
    #text_file = open("./templates/model_performance_analysis.html", "wt",encoding='utf-8')
    #n = text_file.write(example)
    #text_file.close()   
 
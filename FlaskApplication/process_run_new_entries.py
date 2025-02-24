#%load_ext autoreload
#%autoreload 2
import auxiliar_functions
from auxiliar_functions import *

#def new_skills():
#
#    with open('./Transformations/transformer_and_features.pkl', 'rb') as f:
#        loaded_dict = pickle.load(f)
#    
#    transformer = loaded_dict['TFIDFtransformer']
#    feature_names = loaded_dict['TFIDFfeature_names']
#
#    with open('./Transformations/model_selected_features.pkl', 'rb') as f:
#        loaded_dict = pickle.load(f)
#    model_selected_features = loaded_dict['model_selected_features']
#    results = loaded_dict['performance_results']
#
#
#    with open('./Transformations/model_tuning.pkl', 'rb') as f:
#        loaded_dict = pickle.load(f)
#    tuned_model     = loaded_dict['tuned_model']
#
#    with open('./data/newEntries.pkl', 'rb') as f:
#        loaded_dict = pickle.load(f)
#    Xentry     = loaded_dict['skills']
#    #Xentry = Xentry.strip("'\"")
#    Xentry = Xentry.lstrip("'\"").rstrip("'\"")
#    Xentry = Xentry.lstrip('"\"').rstrip('"\"')
#    if type(Xentry)==str:
#        Xentry = pd.DataFrame({'HaveWorkedWith': [Xentry]})
#
#    XentryTFIDF = transformer.transform(Xentry)
#    XentryTFIDF = pd.DataFrame(XentryTFIDF.toarray(), columns=feature_names, index=Xentry.index)
#    XentryTFIDF= XentryTFIDF[model_selected_features['Decision Tree']]
#    prediction = tuned_model.predict(XentryTFIDF)
#    prediction = pd.DataFrame(prediction, columns=['Employed'])
#    result = pd.concat([Xentry.reset_index(), prediction], axis=1)
#    result=result.reset_index(drop=True)
#    result = result.drop('index',axis=1)
#
#    fig = create_table(result,'Employement Prediction',
#        width=900, 
#        height=400,
#        columnwidth=[30,4])
#
#
#    return fig.to_html(full_html=False)

def new_skills():
    # Load necessary data and models
    with open('./Transformations/transformer_and_features.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    transformer = loaded_dict['TFIDFtransformer']
    feature_names = loaded_dict['TFIDFfeature_names']

    with open('./Transformations/model_selected_features.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    model_selected_features = loaded_dict['model_selected_features']

    with open('./Transformations/model_tuning.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    tuned_model = loaded_dict['tuned_model']

    # Load new entries
    with open('./data/newEntries.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    Xentry = loaded_dict['skills']

    # Handle both single entry and multiple entries
    if isinstance(Xentry, str):
        Xentry = Xentry.lstrip("'\"").rstrip("'\"")
        Xentry = Xentry.lstrip('"\"').rstrip('"\"')
        Xentry = pd.DataFrame({'HaveWorkedWith': [Xentry]})
    elif isinstance(Xentry, list):
        Xentry = pd.DataFrame({'HaveWorkedWith': Xentry})
        print(type(Xentry))
    # Transform and predict
    XentryTFIDF = transformer.transform(Xentry)
    XentryTFIDF = pd.DataFrame(XentryTFIDF.toarray(), columns=feature_names, index=Xentry.index)
    XentryTFIDF = XentryTFIDF[model_selected_features['Decision Tree']]
    prediction = tuned_model.predict(XentryTFIDF)
    prediction = pd.DataFrame(prediction, columns=['Employed'])
    result = pd.concat([Xentry.reset_index(), prediction], axis=1)
    result = result.reset_index(drop=True)
    result = result.drop('index', axis=1)

    fig = create_table(result, 'Employment Prediction',
        width=900, 
        height=400,
        columnwidth=[30, 4])

    return fig.to_html(full_html=False)
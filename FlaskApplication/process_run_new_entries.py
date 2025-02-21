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
    skills     = loaded_dict['skills']
    
    print(skills)
    #Xentry = X_testFinal.iloc[0:2,:]
    Xentry='Bash/Shell;HTML/CSS;JavaScript;PHP;Ruby;'
    if type(Xentry)==str:
        Xentry = pd.DataFrame({'HaveWorkedWith': [Xentry]})
    XentryTFIDF = transformer.transform(Xentry)
    XentryTFIDF = pd.DataFrame(XentryTFIDF.toarray(), columns=feature_names, index=Xentry.index)
    XentryTFIDF= XentryTFIDF[model_selected_features['Decision Tree']]
    tuned_model.predict(XentryTFIDF)
 
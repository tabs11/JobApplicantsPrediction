#%load_ext autoreload
#%autoreload 2
import auxiliar_functions
from auxiliar_functions import *


def run_data_transformations():
    loaded_transformer, loaded_feature_names, loaded_Xtfidf_header = load_transformer_and_features()
    
    model_selected_features, results, HyperParameters = model_feature_selection()
    
    fig1 = create_table(loaded_Xtfidf_header, 
             title='TF-IDF features sample (Only 6 out of 115 features shown)', 
             width=1300, 
             height=500
            #columnwidth=[10, 60]
            )

    fig2 = create_table(results, 
             title='Model and Features Selection Results', 
             width=1000, 
             height=250)

    fig3 = create_table(HyperParameters, 
        title='Model tuning', 
        width=500, 
        height=300, 
        columnwidth=[5,10,10])

    fig = make_subplots(
        rows=3, cols=1,
        specs=[[{"type": "table"}], [{"type": "table"}],[{"type": "table"}]],
        vertical_spacing=0.2,  # Increase this value to add more space between plots
        subplot_titles=("TF-IDF features sample (Only 6 out of 115 features shown)", "Model and Features Selection Results", 'Model Tuning')
    )
    
    # Add the first table to the subplot
    fig.add_trace(go.Table(
        header=fig1.data[0].header,
        cells=fig1.data[0].cells
    ), row=1, col=1)
    
    # Add the second table to the subplot
    fig.add_trace(go.Table(
        header=fig2.data[0].header,
        cells=fig2.data[0].cells
    ), row=2, col=1)

    # Add the second table to the subplot
    fig.add_trace(go.Table(
        header=fig3.data[0].header,
        cells=fig3.data[0].cells
    ), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=1400,  # Increase height to accommodate larger titles and spacing
        width=1300,
        title_text="",
        showlegend=False,
        font=dict(size=12),  # Increase base font size,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # Update subplot titles to make them bigger and bold
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=18, color='#204A87', family="Arial, sans-serif", weight="bold")
    
    fig.write_html("./templates/features_transforming.html")
    #    
    import bs4
    with open('./templates/features_transforming.html','r', encoding='utf-8') as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt,features="lxml")
    toreplace='{% extends "login_layout.html" %}\n{% block content %}\n<p>&nbsp;</p>\n<div class="multi_upload_index">\n<h2 style="color:black;background-color:#cccccc" class="text-muted"><b><font size="5">Features Transformation and Model Selection :</font></b></h2>\n<p>&nbsp;</p><table><tr>\n<form class="button" align="right" method="GET" action="/model_performance_analysis" enctype="multipart/form-data">\n<input class="mybutton" type="submit" value="MODEL PERFORMANCE ANALYSIS"  class="span2"></td>\n</form>\n</tr>\n</table>\n<p>&nbsp;</p>\n<table>\n<tr>\n'

    example=str(soup).replace('<html>\n<head><meta charset="utf-8"/></head>\n',toreplace).replace('</html>','{% endblock %}')
    text_file = open('./templates/features_transforming.html', "wt",encoding='utf-8')
    n = text_file.write(example)
    text_file.close()   
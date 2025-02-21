#%load_ext autoreload
#%autoreload 2
import auxiliar_functions
from auxiliar_functions import *

def upload_file():

    path = "./data/stackoverflow_full.csv"
    employmentData = pd.read_csv(path)
    employmentData = employmentData.rename(columns={'Unnamed: 0': 'ID'})
    employmentData = employmentData.dropna()
    employmentData = employmentData[['Employed','HaveWorkedWith']]
    employmentData_head = employmentData.head(10)
    
    fig = create_table(employmentData_head, 
        title='Job Aplicants Data set - New Study', 
        width=1100, 
        height=500,
        columnwidth=[10, 60])

    
    fig.write_html("./templates/plan_analysis.html")
        
    import bs4
    with open('./templates/plan_analysis.html','r', encoding='utf-8') as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt,features="lxml")


    toreplace='{% extends "login_layout.html" %}\n{% block content %}\n<p>&nbsp;</p>\n<div class="multi_upload_index">\n<h2 style="color:black;background-color:#cccccc" class="text-muted"><b><font size="5">Introduction:</font></b></h2>\n<p>&nbsp;</p>\n<table>\n<tr>\n<form class="button" align="right" method="GET" action="/features_transforming" enctype="multipart/form-data">\n<input class="mybutton" type="submit" value="APPLY FEATURES TRANSFORMATION"  class="span2"></td>\n</form>\n</tr>\n</table>\n<p>&nbsp;</p>\n<table>\n<tr>\n'


    example=str(soup).replace('<html>\n<head><meta charset="utf-8"/></head>\n',toreplace).replace('</html>','{% endblock %}')   
    text_file = open("./templates/plan_analysis.html", "wt",encoding='utf-8')
    n = text_file.write(example)
    text_file.close()   



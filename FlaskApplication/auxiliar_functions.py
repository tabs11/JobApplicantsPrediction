# Imports
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate,KFold,cross_val_predict,train_test_split,RandomizedSearchCV,GridSearchCV,learning_curve
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import preprocessing,tree
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score,accuracy_score,precision_score, recall_score, f1_score,roc_auc_score,roc_curve, auc
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from scipy.stats import chi2_contingency,pointbiserialr
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
import math
import pickle
warnings.filterwarnings("ignore")



def custom_tokenizer(text):
    return [token.strip() for token in text.split(';') if token.strip()]
    
def load_transformer_and_features():
    with open('./Transformations/transformer_and_features.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    print("Type of loaded object:", type(loaded_dict))
    
    if isinstance(loaded_dict, dict):
        loaded_transformer = loaded_dict['TFIDFtransformer']
        loaded_feature_names = loaded_dict['TFIDFfeature_names']
        loaded_Xtfidf_header = loaded_dict['TFIDFset_header']
    else:
        print("Loaded object is not a dictionary. It's a:", type(loaded_dict))
        loaded_transformer = loaded_dict
        loaded_feature_names = loaded_transformer.named_transformers_['vectorizer'].get_feature_names_out().tolist()
    
    return loaded_transformer, loaded_feature_names,loaded_Xtfidf_header
    
def model_feature_selection():
    with open('./Transformations/model_selected_features.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    model_selected_features = loaded_dict['model_selected_features']
    results = loaded_dict['performance_results']
    HyperParameters = loaded_dict['HyperParameters']

    return model_selected_features, results, HyperParameters

def model_tuning():
    with open('./Transformations/model_tuning.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    tuned_model     = loaded_dict['tuned_model']
    X_trainNew      = loaded_dict['X_trainNew']
    y_train         = loaded_dict['y_train']
    X_valNew        = loaded_dict['X_valNew']
    y_val           = loaded_dict['y_val']
    X_testNew       = loaded_dict['X_testNew']
    y_test          = loaded_dict['y_test']
    X_test_tfidf    = loaded_dict['X_test_tfidf']
    y_testFinal     = loaded_dict['y_testFinal']
    train_sizes       = loaded_dict['train_sizes']
    train_scores_mean = loaded_dict['train_scores_mean']
    test_scores_mean  = loaded_dict['test_scores_mean']
    perform_metrics   = loaded_dict['df_metrics']
    y_scores          = loaded_dict['y_scores']
    precision_test    = loaded_dict['precision_test']
    recall_test       = loaded_dict['recall_test']
    average_precision_test = loaded_dict['average_precision_test']

    return tuned_model,X_trainNew,y_train,X_valNew,y_val, X_testNew,y_test, X_test_tfidf,y_testFinal, train_sizes, train_scores_mean, test_scores_mean, perform_metrics, y_scores, precision_test, recall_test, average_precision_test

def create_table(data, title='Table', width=1300, height=400, columnwidth=None):
    """
    Create a Plotly table with consistent styling.

    :param data: pandas DataFrame containing the data to be displayed
    :param title: Title of the table
    :param width: Width of the table
    :param height: Height of the table
    :param columnwidth: List of column widths (optional)
    :return: Plotly Figure object
    """

    table_args = dict(
        header=dict(
            values=list(data.columns),
            fill_color='#204A87',
            font=dict(color='white'),
            align='left',
            height=40
        ),
        cells=dict(
            values=[data[col] for col in data.columns],
            fill_color='rgba(0,0,0,0)',
            align='left',
            font=dict(size=14),
            height=30
        )
    )

    if columnwidth is not None:
        table_args['columnwidth'] = columnwidth

    fig = go.Figure(data=[go.Table(**table_args)])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                color='#336699',
                size=20,
                family='Arial',
            ),
            x=0.5,
            xanchor='center',
        ),
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
    )

    return fig


# Compute the correlation matrix
def plot_correlation(data, title='Correlation Heatmap'):
    """
    Create and display an interactive correlation heatmap using Plotly,
    with customized axis colors and matching colorbar font.
    
    Parameters:
    data (pandas.DataFrame): The input dataframe to compute correlations from.
    title (str): The title of the heatmap (default: 'Correlation Heatmap').
    
    Returns:
    plotly.graph_objects.Figure: The Plotly figure object.
    """
    # Compute the correlation matrix
    corr_matrix = data.corr()

    # Define common font properties
    font_color = '#336699'
    font_size = 15

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        zmin=-1,
        zmax=1,
        colorscale='RdBu_r',
        colorbar=dict(
            title='Correlation',
            titleside='right',
            titlefont=dict(color=font_color, size=font_size),
            tickfont=dict(color=font_color, size=font_size)
        ),
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        hoverinfo='text'
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                color=font_color,
                size=20,
                family='Arial',
            ),
            x=0.5,
            xanchor='center',
        ),
        width=800,
        height=700,
        xaxis=dict(
            showgrid=False,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size)
        ),
        yaxis=dict(
            showgrid=False,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size),
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


##feature importance

def plot_feature_importances(model, X, title='Feature Importances'):
    """
    Create and display an interactive bar chart of feature importances using Plotly.
    
    Parameters:
    model: The trained model with feature_importances_ attribute.
    X (pandas.DataFrame): The input dataframe used for training.
    title (str): The title of the chart (default: 'Feature Importances').
    
    Returns:
    plotly.graph_objects.Figure: The Plotly figure object.
    """
    importances = model.feature_importances_
    feature_names = X.columns.tolist()

    # Create a DataFrame for plotting
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df = df.sort_values('Importance', ascending=False)

    # Define common font properties
    font_color = '#336699'
    font_size = 15

    # Bar color
    bar_color = '#336699'

    # Create the bar chart using Plotly Graph Objects
    fig = go.Figure(data=[go.Bar(
        x=df['Feature'],
        y=df['Importance'],
        marker_color=bar_color  # Setting the bar color
    )])

    # Customize the layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                color=font_color,
                size=20,
                family='Arial',
            ),
            x=0.5,
            xanchor='center',
        ),
        xaxis_title='Features',
        yaxis_title='Importance',
        xaxis_tickangle=-45,
        height=750,
        width=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.5,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size)
        ),
        showlegend=False  # Hides the legend, as it's not needed for single-color bars
    )

    return fig






def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    
    font_color = '#336699'
    font_size = 15

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['0', '1'],
        y=['0 ', '1 '],
        colorscale='Blues',
        colorbar=dict(
            title='Count',
            titleside='right',
            titlefont=dict(color=font_color, size=font_size),
            tickfont=dict(color=font_color, size=font_size),
             showticklabels=False
        ),
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverinfo='text'
    ))

    # Add text annotations with custom colors
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if (i == 0 and j == 1) or (i == 1 and j == 0):  # False Positive or False Negative
                text_color = '#336699'
            else:
                text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            
            fig.add_annotation(
                x=['0', '1'][j],
                y=['0 ', '1 '][i],
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color=text_color, size=16)
            )

    # Update layout (rest of the code remains the same)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                color=font_color,
                size=20,
                family='Arial',
            ),
            x=0.5,
            xanchor='center',
        ),
        width=300,
        height=350,
        xaxis=dict(
            title='Predicted',
            showgrid=False,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size)
        ),
        yaxis=dict(
            title='Actual',
            showgrid=False,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size),
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )

    return fig





def compare_confusion_matrixes(*args):
    # Separate data pairs and titles
    data_pairs = [arg for arg in args if isinstance(arg, tuple)]
    titles = [arg for arg in args if isinstance(arg, str)]
    
    # Ensure the number of titles matches the number of data pairs
    if len(titles) != len(data_pairs):
        raise ValueError("The number of titles must match the number of data pairs.")
    
    # Determine the number of confusion matrices
    n = len(data_pairs)
    
    # Calculate the number of rows and columns for the subplot
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    
    # Create subplot titles
    subplot_titles = [
        f"<b><span style='font-size:20px; color:#336699;'>{title}</span></b>"
        for title in titles
    ]
    
    # Create a subplot
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    
    # Create each confusion matrix plot and add it to the subplot
    for i, ((y_true, y_pred), title) in enumerate(zip(data_pairs, titles)):
        row = i // cols + 1
        col = i % cols + 1
        
        cm = plot_confusion_matrix(y_true, y_pred, "")
        fig.add_trace(cm.data[0], row=row, col=col)
        
        # Update axes for each subplot
        fig.update_xaxes(
            title_text="Predicted", 
            title_font=dict(size=17, color='#336699', family='Arial, bold'),
            tickfont=dict(size=17, color='#336699', family='Arial, bold'),
            row=row, col=col
        )
        fig.update_yaxes(
            autorange="reversed",
            title_text="Actual",
            title_font=dict(size=17, color='#336699', family='Arial, bold'),
            tickfont=dict(size=17, color='#336699', family='Arial, bold'), 
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        height=250 * rows, 
        width=200 * cols,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Show the figure
    return fig





##precision recall
def plot_precision_recall_curve(precision, recall, average_precision, title='Precision-Recall Curve (Test Set)'):
    # Define common font properties
    font_color = '#336699'
    font_size = 15

    # Line color
    line_color = '#336699'

    # Create Plotly figure
    fig = go.Figure()

    # Add trace for Precision-Recall curve
    fig.add_trace(go.Scatter(
        x=recall, 
        y=precision,
        mode='lines',
        name=f'Test AP={average_precision:.2f}',
        line=dict(color=line_color, width=2)
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                color=font_color,
                size=20,
                family='Arial',
            ),
            x=0.5,
            xanchor='center',
        ),
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=800,
        width=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size),
            range=[0, 1.1]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.5,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size),
            range=[0, 1]
        ),
        legend=dict(
            font=dict(color=font_color, size=font_size),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            visible=True  # Explicitly set the legend to be visible

        ),
        showlegend=True
    )
    return fig


#learning curve
def plot_learning_curve(train_sizes, train_scores_mean, test_scores_mean, title='Learning Curves'):
    # Define common font properties
    font_color = '#336699'
    font_size = 15

    # Create Plotly figure
    fig = go.Figure()

    # Add traces for training and cross-validation scores
    fig.add_trace(go.Scatter(
        x=train_sizes, 
        y=train_scores_mean,
        mode='lines',
        name='Training score',
        line=dict(color='#336699', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes, 
        y=test_scores_mean,
        mode='lines',
        name='Cross-validation score',
        line=dict(color='#FF6B6B', width=2)
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=font_color, size=20, family='Arial'),
            x=0.5,
            xanchor='center',
        ),
        xaxis_title='Training examples',
        yaxis_title='Score',
        height=800,
        width=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.5,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size)
        ),
        legend=dict(
            font=dict(color=font_color, size=font_size),
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig


## roc curve
def plot_roc_curve(fpr, tpr, roc_auc, title='Receiver Operating Characteristic (ROC) Curve'):
    # Define common font properties
    font_color = '#336699'
    font_size = 15

    # Create Plotly figure
    fig = go.Figure()

    # Add trace for ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(color='#336699', width=2)
    ))

    # Add trace for random classifier
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=font_color, size=20, family='Arial'),
            x=0.5,
            xanchor='center',
        ),
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=800,
        width=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.5,
            color=font_color,
            tickfont=dict(color=font_color, size=font_size)
        ),
        legend=dict(
            font=dict(color=font_color, size=font_size),
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig
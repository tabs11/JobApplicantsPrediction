#%load_ext autoreload
#%autoreload 2
import auxiliar_functions
from auxiliar_functions import *

def model_performance_analysis():
    tuned_model,X_trainNew,y_train,X_valNew,y_val, X_testNew,y_test, X_test_tfidf,y_testFinal, train_sizes, train_scores_mean, test_scores_mean, perform_metrics, y_scores, precision_test, recall_test, average_precision_test = model_tuning()

    y_pred_train_cv = cross_val_predict(tuned_model, X_trainNew, y_train, cv=10)
    # For validation set
    y_pred_val_cv = cross_val_predict(tuned_model, X_valNew, y_val, cv=10)
    # For test set
    y_pred_test_cv = cross_val_predict(tuned_model, X_testNew, y_test, cv=10)
    # For final test set
    y_pred_testFinal_cv = cross_val_predict(tuned_model, X_test_tfidf, y_testFinal, cv=10)

    fpr, tpr, _ = roc_curve(np.array(y_test).astype(int), y_scores)
    roc_auc = auc(fpr, tpr)

    

    fig1 = plot_feature_importances(tuned_model, X_trainNew,title = "Features Importance after tuning")
    fig2 = plot_learning_curve(train_sizes, train_scores_mean, test_scores_mean,title = "Learning Curve")
    fig3 = compare_confusion_matrixes(
    (np.array(y_train).astype(int), y_pred_train_cv), "Confusion Matrix - Training Set",
    (np.array(y_val).astype(int), y_pred_val_cv), "Confusion Matrix - Validation Set",
    (np.array(y_test).astype(int), y_pred_test_cv), "Confusion Matrix - Test Set",
    (np.array(y_testFinal).astype(int), y_pred_testFinal_cv), "Confusion Matrix - Final Test Set"
    )
    fig4 = plot_precision_recall_curve(precision_test, recall_test, average_precision_test)
    fig5 = plot_roc_curve(fpr, tpr, roc_auc)
    fig6 = create_table(perform_metrics,'Perfomance Metrics',
        width=900, 
        height=400,
        columnwidth=[10,10,10,10,10,10])

    fig = make_subplots(
        rows=6, cols=2,  # Changed from 5 to 6 rows
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "scatter"}],  # New row for Precision-Recall and ROC curves
            [{"type": "table", "colspan": 2}, None],
            [None, None]  # Extra row to maintain layout
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=("Features Importance", "Learning Curve", 
                        "Confusion Matrix - Training Set", "Confusion Matrix - Validation Set",
                        "Confusion Matrix - Test Set", "Confusion Matrix - Final Test Set",
                        f"Precision-Recall Curve<br><sub>{fig4.data[0].name}</sub>", 
                        f"Receiver Operating Characteristic (ROC) Curve<br><sub><span style='color:#336699'>{fig5.data[0].name}</span><br><span style='color:#FF6B6B'>Random Classifier</span></sub>",
                        'Performance Metrics', '')
    )
    # Add the bar plot to the subplot (Features Importance)
    fig.add_trace(go.Bar(
        x=fig1.data[0].x,
        y=fig1.data[0].y,
        marker=fig1.data[0].marker,
        showlegend=False
    ), row=1, col=1)
    
    # Add the scatter plot (Learning Curve)
    fig.add_trace(go.Scatter(
        x=fig2.data[0].x,
        y=fig2.data[0].y,
        mode='lines',
        name='Training score',
        line=dict(color='#336699', width=2)
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=fig2.data[1].x,
        y=fig2.data[1].y,
        mode='lines',
        name='Cross-validation score',
        line=dict(color='#FF6B6B', width=2)
    ), row=1, col=2)
    
    # Add confusion matrix plots
    for i, (y_true, y_pred) in enumerate([
        (np.array(y_train).astype(int), y_pred_train_cv),
        (np.array(y_val).astype(int), y_pred_val_cv),
        (np.array(y_test).astype(int), y_pred_test_cv),
        (np.array(y_testFinal).astype(int), y_pred_testFinal_cv)
    ]):
        cm = confusion_matrix(y_true, y_pred)
        row = 2 if i < 2 else 3
        col = 1 if i % 2 == 0 else 2
        
        fig.add_trace(go.Heatmap(
            z=cm,
            x=['0', '1'],
            y=['0', '1'],
            colorscale='Blues',
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverinfo='text'
        ), row=row, col=col)
    
    # Add Precision-Recall curve
    fig.add_trace(go.Scatter(
        x=fig4.data[0].x,
        y=fig4.data[0].y,
        mode='lines',
        name=fig4.data[0].name,
        showlegend=False,
        line=dict(color='#336699', width=2)
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=fig5.data[0].x,
        y=fig5.data[0].y,
        mode='lines',
        name=fig5.data[0].name,
        showlegend=False,
        line=dict(color='#336699', width=2)
    ), row=4, col=2)

    # Add random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        showlegend=False,
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ), row=4, col=2)
    
    fig.add_trace(go.Table(
        header=fig6.data[0].header,
        cells=fig6.data[0].cells
    ), row=5, col=1)    
    # Update layout
    fig.update_layout(
    height=2400,
    width=1200,
    title_text="",
    showlegend=True,
    font=dict(size=12),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(
        yanchor="top",
        y=0.99,  # Changed from 0.01 to 0.99
        xanchor="right",
        x=0.99,
        orientation="h"  # Added to make the legend horizontal
    )
)
    
    # Update subplot titles
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=18, color='#204A87', family="Arial, sans-serif", weight="bold")
    
    # Update x-axis and y-axis labels
    fig.update_xaxes(title_text="Training examples", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    
    # Update x-axis and y-axis labels for Precision-Recall curve
    fig.update_xaxes(title_text="Recall", row=4, col=1)
    fig.update_yaxes(title_text="Precision", row=4, col=1)

    # Update x-axis and y-axis labels for ROC curve
    fig.update_xaxes(title_text="False Positive Rate", row=4, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=4, col=2)
    
    # Update axes for confusion matrices
    for i in range(4):
        row = 2 if i < 2 else 3
        col = 1 if i % 2 == 0 else 2
        fig.update_xaxes(title_text="Predicted", row=row, col=col)
        fig.update_yaxes(title_text="Actual", autorange="reversed", row=row, col=col)
     
    
    fig.write_html("./templates/model_performance_analysis.html")
        
    import bs4
    with open('./templates/model_performance_analysis.html',"r",encoding='utf-8') as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt,features="lxml")
    toreplace='{% extends "login_layout.html" %}\n{% block content %}\n</form>\n<p>&nbsp;</p>\n<div class="multi_upload_index">\n<h2 style="color:black;background-color:#cccccc" class="text-muted"><b><font size="5">MODEL PERFORMANCE ANALYSIS:</font></b></h2>\n'

    example=str(soup).replace('<html>\n<head><meta charset="utf-8"/></head>\n',toreplace).replace('</html>','{% endblock %}')
    text_file = open("./templates/model_performance_analysis.html", "wt",encoding='utf-8')
    n = text_file.write(example)
    text_file.close()   
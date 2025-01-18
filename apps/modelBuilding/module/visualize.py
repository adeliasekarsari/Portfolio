import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def plot_perbandingan(df_summ, col_main, col1, col2, sortby, plot_height=600):
    # Data prep
    cluster =  df_summ.sort_values(sortby, ascending = False)[col_main].values
    total_comp = df_summ.sort_values(sortby, ascending = False)[col1].values
    min_dist_comp = df_summ.sort_values(sortby, ascending = False)[col2].values

    fig = go.Figure(
        data=go.Bar(
            x=cluster,
            y=total_comp,
            name=col1,
            marker=dict(color='#B0C2E1'),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cluster,
            y=min_dist_comp,
            yaxis="y2",
            name=col2,
            marker=dict(color="darkblue"),
        )
    )
    
    col_name = col1
    fig.update_layout(
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text=col_name),
            side="left",
        ),
        yaxis2=dict(
            title=dict(text=col2),
            side="right",
            overlaying="y",
            tickmode="sync",
        ),
        height = plot_height
    )
    fig.update_layout(title=f'{col1} and {col2}')
    return fig

def plot_histogram(dataframe, var):  
    """
    dataframe:
    """
    fig = px.histogram(dataframe[dataframe[var]!=0], 
                            x=var,
                            title = f'Histogram based on {var}'
                            )
    fig.update_layout(bargap=0.2)
    
    fig.update_traces(marker_color='#B0C2E1', 
                marker_line_width=1.5, opacity=1, )
    return fig

def plot_scatter(dataframe, unique_col, y1, y2):
    fig = px.scatter(dataframe, 
                        x=y1, 
                        y=y2, 
                        hover_data=[unique_col],trendline="ols")
    fig.update_layout(title=f'Scatter Plot Correlation with Correlation : {dataframe[[y1,y2]].corr()[y1][y2].round(3)}')
    return fig

def heatmap_plot(correlation):
    fig = px.imshow(np.array(correlation),
                labels=dict(x="Parameters", 
                            y='Y', 
                            color='Correlation'),
                x=correlation.columns.tolist(),
                y=correlation.index.tolist(),zmin = -1, zmax = 1,
                height=correlation.shape[0]*100,
            text_auto=True, 
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    title = f'Correlation'
                )
    fig.update_layout(
            margin=dict(l=150, r=100, t=100, b=100),
            yaxis = dict( tickfont = dict(size=16)),
            xaxis = dict( tickfont = dict(size=16))
        )
    fig.update_traces(textfont_size=16)
    fig.update_xaxes(side="top")
    return fig

def correlation_bar_chart(correlation_, corr_):
    fig = px.bar(correlation_.sort_values(corr_), 
                    y='index', 
                    x=corr_,
                    orientation='h',
                    color='Keterangan',
                    text = corr_,
                    color_discrete_sequence=correlation_.sort_values(corr_)['Color'].unique().tolist(),
                    
                    )
    fig.update_layout(
        autosize=False,
        width=800,
        height=correlation_.shape[0]*50,
        font=dict(
        size=16,  # Set the font size here
        color="RebeccaPurple"),
        barmode='stack', title = f'Correlation {corr_}',
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='inside',
        showline=True,
        linecolor='black',
        gridcolor='white'
    )
    fig.update_layout( yaxis = dict( tickfont = dict(size=16)))
    return fig

def voi_chart(df_shap):
    df_shap = df_shap.sort_values('abs', ascending = True).drop_duplicates("feature")
    if df_shap.shape[0]>15:
        height = df_shap.shape[0]*30
    elif df_shap.shape[0]>7:
        height = 600
    else :
        height = 300
        
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name='Importance Score',
            x=df_shap['abs'],
                orientation='h',
            y=df_shap['feature'],
            marker_color=df_shap['Color']))
    fig.update_layout(
        autosize=False,
        width=800,
        height= height,
        barmode='stack', title = 'Feature Importance'
    )
    return fig
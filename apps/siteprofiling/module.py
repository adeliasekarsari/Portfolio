import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def demog_viz(df, color):
    fig = px.bar(df, 
                x='Ages', 
                y='population',
                hover_data=['population','Ages'], 
                color = 'Ages', 
                color_discrete_sequence=color, 
                height=500,
                animation_frame="dt", 
                )

    for step in fig.layout.sliders[0].steps:
        step["args"][1]["frame"]["redraw"] = True
    fig.update_xaxes(type='category')
    fig.update_layout(yaxis_range=[0,round(df['population'].max()/1000)*1000])
    return fig

def ses_viz(df):
    fig = px.bar(df, 
                x='nilai', 
                y='ses',
                hover_data=['nilai'], 
                color = 'nilai', 
                color_discrete_map={
                                    'high':'#0D2C5B',
                                    'medium-high':'#206DAB',
                                    'medium-low':'#FDD212',
                                    'low':'#FDF8B3'}, 
                labels={
                        "ses": "Percentage (%)",
                        "nilai": "SocioEconomic Status",},
                height=500,
                animation_frame="dt", 
                category_orders={'nilai':['high','medium-high','medium-low','low']}
                )
    fig.update_xaxes(categoryorder='array', 
                        categoryarray= ['high','medium-high','medium-low','low'])
    fig.update_layout(yaxis_range=[0,100])
    for step in fig.layout.sliders[0].steps:
        step["args"][1]["frame"]["redraw"] = True
    fig.update_xaxes(type='category')

    return fig

def poi_viz(df, color):
    fig = px.bar(df.sort_values(['range_area','id_merchant']), 
                y='id_merchant', 
                x='nama_kategori',
                hover_data=['id_merchant'], 
                color = 'nama_kategori', 
                color_discrete_sequence=color, 
                labels={
                        "id_merchant": "Total POI",
                        "nama_kategori": "Category Names",},
                height=500,
                width = 900,
                animation_frame="range_area", 
                )
    fig.update_xaxes(type='category')
    fig.update_layout(yaxis_range=[0,round(df['id_merchant'].max())],
                      xaxis = None,
                    title = 'Total POI based on Category')
    for step in fig.layout.sliders[0].steps:
        step["args"][1]["frame"]["redraw"] = True
    fig.update_xaxes(type='category')
    return fig

def demog_viz_dis(df, color):
    fig = px.bar(df, 
                x='Ages', 
                y='population',
                hover_data=['population','Ages'], 
                color = 'Ages', 
                color_discrete_sequence=color, 
                height=500,
                )

    fig.update_xaxes(type='category')
    fig.update_layout(yaxis_range=[0,round(df['population'].max()/1000)*1000])
    return fig

def ses_viz_dis(df):
    fig = px.bar(df, 
                x='nilai', 
                y='ses',
                hover_data=['nilai'], 
                color = 'nilai', 
                color_discrete_map={
                                    'high':'#0D2C5B',
                                    'medium-high':'#206DAB',
                                    'medium-low':'#FDD212',
                                    'low':'#FDF8B3'}, 
                labels={
                        "ses": "Percentage (%)",
                        "nilai": "SocioEconomic Status",},
                height=500,
                category_orders={'nilai':['high','medium-high','medium-low','low']}
                )
    fig.update_xaxes(categoryorder='array', 
                        categoryarray= ['high','medium-high','medium-low','low'])
    fig.update_layout(yaxis_range=[0,100])
    fig.update_xaxes(type='category')

    return fig

def poi_viz_dis(df, color):
    fig = px.bar(df.sort_values(['range_area','id_merchant']), 
                y='id_merchant', 
                x='nama_kategori',
                hover_data=['id_merchant'], 
                color = 'nama_kategori', 
                color_discrete_sequence=color, 
                labels={
                        "id_merchant": "Total POI",
                        "nama_kategori": "Category Names",},
                height=500,
                width = 900,
                )
    fig.update_xaxes(type='category')
    fig.update_layout(yaxis_range=[0,round(df['id_merchant'].max())],
                      xaxis = None,
                    title = 'Total POI based on Category')
    fig.update_xaxes(type='category')
    return fig

#Visualisasi rata-rata pengunjung setiap jamnya
def mw_hour(mw):
    mw_group_hour = mw.groupby('hour')['device_id'].nunique().reset_index()
    mw_group_hour.columns = ['hour','total_device']
    fig = px.line(mw_group_hour, 
                  x="hour", 
                  y="total_device", 
                  height = 350,
                 labels={
                         "hour": "Hours",
                         "total_device": "Total Device",})
    fig.update_traces(textposition="bottom right")
    fig.update_traces(line_color='lightblue',
                     line_width = 6,)
    fig.update_layout(font = dict(
                size=21),
                title = 'Total Visitor per Hour based on Telco Data')
    return fig

def mw_chart_weekdays(mw):
    #Weekdays
    hour = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    mw_weekday = mw.groupby('WeekDay')['device_id'].nunique().reset_index()
    df1 = mw_weekday.set_index('WeekDay')
    mw_weekday = df1.loc[hour].reset_index()
    mw_weekday.rename(columns = {'device_id':'Total Device'}, inplace = True)
    fig = px.line(mw_weekday, 
                 x='WeekDay',
                 y='Total Device',
                 height = 350,
                 width = 700,
                title = "Total Visitor Weekdays based on Telco Data")
    fig.update_xaxes(categoryorder='array', 
                     categoryarray= [str(x) for x in hour])
    fig.update_traces(line_color='lightblue',
                     line_width = 6,marker_color='lightblue', 
                          marker_line_color='#FDD212',
                          marker_line_width=0.4)
    fig.update_layout(xaxis_title='WeekDays', 
                      yaxis_title="Total Population",
                      font = dict(
                size=21))
    return fig

def mw_chart_heatmap(mw):
    color1 = '#325285'
    color2 = '#FDD212'
    #Heatmap by days and Hour
    test = mw.copy()
    test = test.groupby(['WeekDay','hour'])['device_id'].nunique().reset_index()
    data_ = pd.pivot(test, index = 'WeekDay', columns = 'hour').reset_index()
    hour = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df1 = data_.set_index('WeekDay')
    data_ = df1.loc[hour].reset_index()
    list_all = []
    for i in range(0, len(data_)):
        list_ = data_[data_.index==i].drop(columns=['WeekDay']).fillna(0).T[i].tolist()
        list_all.append(list_)
    fig = px.imshow(list_all, 
                    y = hour,
                    x = [str(x) for x in range(mw.hour.min(),mw.hour.min()+len(list_all[0]))],
                    text_auto=".2f", 
                    color_continuous_scale='Blues', 
                    aspect="auto",
                    height = 500,
                    width = 1200,
                    title ='Telco Data Heatmap',)
    fig.update_layout(
        yaxis_title="WeekDays", 
        xaxis_title="Hours",
        font = dict(
                size=21))
    return fig
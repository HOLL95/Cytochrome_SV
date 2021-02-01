import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np
from dash.dependencies import Output, Input
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
true_error=2.8788229236284884
curve_data=np.load("Likelihood_curves.npy", allow_pickle=True).item()
surface_data=np.load("2D_likelihoods.npy", allow_pickle=True).item()
true_params=[-0.07133935323836932, 0.04433940419884379, 217.58306150192792, 135.0495596023161, 9.62463515705647e-06, 0.01730398477905308, 0.04999871276633058, -0.0007206743270165433, 1.37095896576959e-11, 4.7220768639743085, 4.554136092744141, 0.5999999989106146]
parameters=["E0_mean","E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","CdlE3","gamma","cap_phase","phase", "alpha"]
true_param_dict=dict(zip(parameters, true_params))
total_names={
    "E_0": 'Midpoint potential',
    'E_start': 'Start potential', #(starting dc voltage - V)
    'E_reverse': 'Reverse potential',
    'omega':'Potential frequency',#8.88480830076,  #    (frequency Hz)
    'd_E': "Potential amplitude",   #(ac voltage amplitude - V) freq_range[j],#
    'v': "Scan rate",   #       (scan rate s^-1)
    'area': "Area", #(electrode surface area cm^2)
    'Ru': "Uncompensated resistance",  #     (uncompensated resistance ohms)
    'Cdl': "Linear double-layer capacitance", #(capacitance parameters)
    'CdlE1': "1st order capacitance",#0.000653657774506,
    'CdlE2': "2nd order capacitance",#0.000245772700637,
    'CdlE3': "3rd order capacitance",#1.10053945995e-06,
    'gamma': 'Surface coverage',
    'k_0': 'Rate constant', #(reaction rate s-1)
    'alpha': "Symmetry factor",
    "E0_mean":"Midpoint potential mean",
    "E0_std": "Midpoint potential standard deviation",
    "E0_skew": "Midpoint potential skew",
    "cap_phase":"Capacitance phase",
    'phase' : "Phase",
}
unit_dict={
    "E_0": "V",
    'E_start': "V", #(starting dc voltage - V)
    'E_reverse': "V",
    'omega':"Hz",#8.88480830076,  #    (frequency Hz)
    'd_E': "V",   #(ac voltage amplitude - V) freq_range[j],#
    'v': '$s^{-1}$',   #       (scan rate s^-1)
    'area': '$cm^{2}$', #(electrode surface area cm^2)
    'Ru': "Ohms",  #     (uncompensated resistance ohms)
    'Cdl': "F", #(capacitance parameters)
    'CdlE1': "",#0.000653657774506,
    'CdlE2': "",#0.000245772700637,
    'CdlE3': "",#1.10053945995e-06,
    'gamma': 'mol cm^-2',
    'k_0': 's^-1', #(reaction rate s-1)
    'alpha': "",
    "E0_mean":"V",
    "E0_std": "V",
    "E0_skew":"",
    "k0_shape":"",
    "k0_loc":"",
    "k0_scale":"",
    "cap_phase":"rads",
    'phase' : "rads",
    "alpha_mean": "",
    "alpha_std": "",
    "":"",
    "noise":"",
    "error":"$\\mu A$",
}
app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='top_curve_x_axis',
                options=[{'label': total_names[x], 'value': x} for x in parameters],
                value='E0_mean'
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='bottom_curve_x_axis',
                options=[{'label': total_names[x], 'value': x} for x in parameters],
                value='E0_std'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
                id='top_curve',
        ),dcc.Graph(id='surface'),
    ], style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='bottom_curve'),

    ], style={'display': 'inline-block', 'width': '49%'}),


])


@app.callback(
    [Output('top_curve', 'figure'),Output('bottom_curve', 'figure'),Output('surface', 'figure')],
    [Input('top_curve_x_axis', 'value'),
     Input('bottom_curve_x_axis', 'value')])
def update_graph(xaxis_name, yaxis_name):
    surface_name=xaxis_name+"_"+yaxis_name

    if surface_name not in surface_data.keys():
        surface_name=yaxis_name+"_"+xaxis_name
        ylabel=xaxis_name
        xlabel=yaxis_name
    else:
        ylabel=yaxis_name
        xlabel=xaxis_name
    top_curve=go.Figure(data={"x":curve_data[xaxis_name]["X"], "y":curve_data[xaxis_name]["Y"]},
                        layout={'xaxis':{"title":{"text":total_names[xaxis_name]+"("+unit_dict[xaxis_name]+")"}},
                        'yaxis':{"title":{"text":"RMSE"}}}
                        )
    bottom_curve=go.Figure(data={"x":curve_data[yaxis_name]["Y"], "y":curve_data[yaxis_name]["X"]},
                        layout={'yaxis':{"title":{"text":total_names[yaxis_name]+"("+unit_dict[yaxis_name]+")"}},
                        'xaxis':{"title":{"text":"RMSE"}}}
                        )
    if xaxis_name==yaxis_name:
        surface=top_curve
    else:
        print(surface_name)

        surface=go.Figure(data=[go.Surface(z=surface_data[surface_name]["Z"], x=surface_data[surface_name]["X"], y=surface_data[surface_name]["Y"])])
        surface.update_layout(scene={"xaxis_title":xlabel+"("+unit_dict[xlabel]+")",
        "yaxis_title":ylabel+"("+unit_dict[ylabel]+")",
        "zaxis_title":"RMSE"})
        surface.add_trace(go.Scatter3d(x=[true_param_dict[xlabel]], y=[true_param_dict[ylabel]], z=[true_error], marker=dict(size=4)))

    return [top_curve, bottom_curve, surface]









if __name__ == '__main__':
    app.run_server(debug=True)

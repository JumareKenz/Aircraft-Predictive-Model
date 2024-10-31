import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import numpy as np
from io import StringIO
import base64
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the pre-trained model and preprocessing pipeline
model = joblib.load("final_model.joblib")
full_pipeline = joblib.load("full_pipeline.joblib")

# Define feature columns used in model training
columns = [
    'id', 'cycle', 'op_setting1', 'op_setting2', 'op_setting3', 's1', 's2', 's3', 's4',
    's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
    's17', 's18', 's19', 's20', 's21'
]
features = [
    'op_setting1', 'op_setting2', 's2', 's3', 's4', 's7', 's8', 's9', 's11',
    's12', 's13', 's14', 's15', 's17', 's20', 's21'
]

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Aircraft RUL Prediction & Evaluation"), className="text-center")),
    
    # Tabs for data input and RUL comparison
    dbc.Row([
        dbc.Col(html.Div("Select Input Method"), width=12, className="text-center"),
        dbc.Col(dcc.Tabs([
            dcc.Tab(label="Upload Test Data", children=[
                dcc.Upload(
                    id='upload-test-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select Test Data File')]),
                    style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                           'borderWidth': '1px', 'borderStyle': 'dashed',
                           'borderRadius': '5px', 'textAlign': 'center'},
                    multiple=False
                )
            ]),
            dcc.Tab(label="Manual Input", children=[
                html.Div([
                    html.Div([
                        html.Label(col),
                        dcc.Input(id=f'input-{col}', type="number", placeholder=f"Enter {col}", step=0.01)
                    ], style={'padding': '5px'}) for col in features
                ]),
                html.Button("Predict RUL", id="predict-button", className="btn btn-primary mt-2")
            ]),
            dcc.Tab(label="Upload Actual RUL File", children=[
                dcc.Upload(
                    id='upload-rul-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select RUL File')]),
                    style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                           'borderWidth': '1px', 'borderStyle': 'dashed',
                           'borderRadius': '5px', 'textAlign': 'center'},
                    multiple=False
                ),
                html.Div(id="metrics-output", className="mt-4")
            ])
        ]))
    ]),
    dbc.Row(dbc.Col(html.Div(id='output-data-upload'))),
], fluid=True)

# Parse test data file
def parse_test_data(contents):
    content_type, content_string = contents.split(',')
    decoded = StringIO(base64.b64decode(content_string).decode('utf-8'))
    df_test_data = pd.read_csv(decoded, sep=" ", header=None)
    
    # Drop last two columns and rename to expected format
    df_test_data = df_test_data.iloc[:, :-2]
    df_test_data.columns = columns[:len(df_test_data.columns)]
    df_test_features = df_test_data[features]
    return df_test_features

# Parse RUL data file
def parse_rul_data(contents):
    content_type, content_string = contents.split(',')
    decoded = StringIO(base64.b64decode(content_string).decode('utf-8'))
    rul_data = pd.read_csv(decoded, header=None, names=["Actual RUL"])
    return rul_data

@app.callback(
    Output('output-data-upload', 'children'),
    Output('metrics-output', 'children'),
    Input('upload-test-data', 'contents'),
    Input('upload-rul-data', 'contents'),
    Input("predict-button", "n_clicks"),
    [State(f'input-{col}', 'value') for col in features]
)
def update_output(test_data_contents, rul_data_contents, predict_click, *manual_inputs):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div(["Awaiting input..."]), ""
    
    if ctx.triggered[0]["prop_id"] == "upload-test-data.contents":
        data = parse_test_data(test_data_contents)
        if isinstance(data, str):
            return html.Div([f"Error processing file: {data}"]), ""
    else:
        # Manual input mode
        manual_values = list(manual_inputs)
        if any(v is None for v in manual_values):
            return html.Div(["Please fill in all input fields for manual prediction."]), ""
        
        data = pd.DataFrame([manual_values], columns=features)
    
    # Preprocess the data and predict
    try:
        prepared_data = full_pipeline.transform(data)
        predictions = model.predict(prepared_data)
        predictions_df = pd.DataFrame({"Predicted RUL": predictions})

        # Compare with actual RUL if provided
        if rul_data_contents:
            rul_data = parse_rul_data(rul_data_contents)
            if len(predictions_df) != len(rul_data):
                return html.Div(["Prediction and actual RUL data lengths do not match."]), ""
            
            predictions_df["Actual RUL"] = rul_data["Actual RUL"]
            # Calculate evaluation metrics
            mae = mean_absolute_error(predictions_df["Actual RUL"], predictions_df["Predicted RUL"])
            rmse = mean_squared_error(predictions_df["Actual RUL"], predictions_df["Predicted RUL"]) ** 0.5
            r2 = r2_score(predictions_df["Actual RUL"], predictions_df["Predicted RUL"])

            metrics_output = html.Div([
                html.H5("Evaluation Metrics"),
                html.P(f"Mean Absolute Error (MAE): {mae:.2f}"),
                html.P(f"Root Mean Squared Error (RMSE): {rmse:.2f}"),
                html.P(f"R² Score: {r2:.2f}")
            ])
        else:
            metrics_output = ""

        # Display predictions
        return html.Div([
            html.H5("Predicted Remaining Useful Life (RUL)"),
            dbc.Table.from_dataframe(predictions_df, striped=True, bordered=True, hover=True)
        ]), metrics_output

    except Exception as e:
        return html.Div([f"An error occurred: {e}"]), ""


if __name__ == '__main__':
    app.run_server(debug=True)

# dashboard_drogas_corregido.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
import numpy as np
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Cargar y preparar datos
df = pd.read_excel('dataset/Drug_addicion_in_spain.xlsx')

# Limpiar espacios vacíos
df = df.applymap(lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x)

# Traducir nombres de columnas
column_translations = {
    'Age': 'Edad',
    'Gender': 'Género',
    'Education': 'Educación',
    'Enjoyable with-': 'Disfrutable con-',
    'Live with': 'Vive con',
    'Conflict with law': 'Conflicto con la ley',
    'Motive about drug': 'Motivo sobre las drogas',
    'Spend most time': 'Pasa la mayor parte del tiempo',
    'Failure in life': 'Fracaso en la vida',
    'Mental/emotional problem': 'Problema mental/emocional',
    'Suicidal thoughts': 'Pensamientos suicidas',
    'Family relationship': 'Relación familiar',
    'Financials of family': 'Finanzas familiares',
    'Addicted person in family': 'Persona adicta en la familia',
    'no. of friends': 'Número de amigos',
    'Withdrawal symptoms': 'Síntomas de abstinencia',
    'friends\' houses at night': 'Casas de amigos por la noche',
    'Satisfied with workplace': 'Satisfecho con el lugar de trabajo',
    'Case in court': 'Caso en la corte',
    'Living with drug user': 'Vive con consumidor de drogas',
    'Smoking': 'Fumador',
    'Ever taken drug': 'Alguna vez ha consumido drogas',
    'Friends influence': 'Influencia de amigos',
    'If chance given to taste drugs': 'Si se le da la oportunidad de probar drogas',
    'Easy to control use of drug': 'Facilidad para controlar el uso de drogas',
    'Frequency of drug usage': 'Frecuencia de uso de drogas'
}

df = df.rename(columns=column_translations)

# Manejar valores nulos - usar 'unknown' en lugar de "Unknown"
for col in df.columns:
    df[col] = df[col].fillna('unknown')

# Normalizar texto (minúsculas y sin espacios extras)
for col in df.columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Normalizar valores de género a español
if 'Género' in df.columns:
    genero_map = {
        'female': 'mujer',
        'femala': 'mujer',
        'male': 'hombre',
        'mle': 'hombre'
    }
    df['Género'] = df['Género'].replace(genero_map)

# Normalizar valores de problemas mentales/emocionales a español
if 'Problema mental/emocional' in df.columns:
    problemas_map = {
        'depression': 'depresión',
        'depressed': 'depresión',
        'anxiety': 'ansiedad',
        'stress': 'estrés',
        'mental stress': 'estrés mental',
        'emotional stress': 'estrés emocional',
        'panic attack': 'ataque de pánico',
        'insomnia': 'insomnio',
        'sleeping problem': 'problemas de sueño',
        'sleep problem': 'problemas de sueño',
        'loneliness': 'soledad',
        'aggressive': 'agresividad',
        'anger': 'ira',
        'frustration': 'frustración',
        'guilt': 'culpa',
        'low self esteem': 'baja autoestima',
        'lack of confidence': 'falta de confianza',
        'trauma': 'trauma',
        'bipolar': 'bipolaridad',
        'schizophrenia': 'esquizofrenia',
        'ocd': 'trastorno obsesivo compulsivo',
        'phobia': 'fobia',
        'tension': 'tensión',
        'inferiority': 'inferioridad',
        'others': 'otros',
        'none': 'ninguno'
    }
    df['Problema mental/emocional'] = df['Problema mental/emocional'].replace(problemas_map)

    # Reemplazos parciales para combinaciones tipo "tension/anxiety"
    reemplazos_regex = [
        (r'\btension\b', 'tensión'),
        (r'\banxiet\w*\b', 'ansiedad'),
        (r'\bdepress\w*\b', 'depresión'),
        (r'\binferiority\b', 'inferioridad'),
        (r'\bguilt\b', 'culpa'),
        (r'\banger\b', 'ira'),
        (r'\bothers\b', 'otros'),
        (r'\bnone\b', 'ninguno')
    ]
    for patron, reemplazo in reemplazos_regex:
        df['Problema mental/emocional'] = df['Problema mental/emocional'].str.replace(
            patron, reemplazo, regex=True
        )
    df['Problema mental/emocional'] = df['Problema mental/emocional'].str.replace('/', ' / ', regex=False)

print("Datos cargados correctamente!")
print(f"Total de registros: {len(df)}")
print(f"Columnas disponibles: {df.columns.tolist()[:10]}...")  # Mostrar solo las primeras 10
print(f"Resumen de datos:")
print(f"   - Edades únicas: {df['Edad'].nunique()}")
print(f"   - Géneros únicos: {df['Género'].nunique()}")

# Crear variable objetivo para análisis predictivo
df['consumidor_riesgo'] = np.where(df['Alguna vez ha consumido drogas'].str.contains('yes'), 1, 0)

# Entrenar modelo predictivo (probabilidad de consumo)
feature_cols = [
    'Género',
    'educacion_categoria',
    'frecuencia_categoria',
    'Influencia de amigos',
    'Fumador',
    'Conflicto con la ley',
    'Vive con consumidor de drogas',
    'Problema mental/emocional'
]
feature_cols = [col for col in feature_cols if col in df.columns]
model_trained = False
model_pipeline = None

if feature_cols:
    try:
        X = df[feature_cols].copy()
        y = df['consumidor_riesgo'].astype(int)

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols)
            ]
        )

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                class_weight='balanced'
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model_pipeline.fit(X_train, y_train)
        model_trained = True
    except Exception as e:
        print(f"Error entrenando modelo predictivo: {e}")

# Definir categorías para variables clave con manejo de errores
def categorizar_edad(edad):
    try:
        if isinstance(edad, str):
            if '15 to 22' in edad:
                return '15-22 años'
            elif '22 to 35' in edad:
                return '22-35 años'
            elif '35 to 45' in edad:
                return '35-45 años'
            elif '45 above' in edad:
                return '45+ años'
        return 'Desconocido'
    except:
        return 'Desconocido'

def categorizar_educacion(educacion):
    try:
        if isinstance(educacion, str):
            if 'undergraduate' in educacion:
                return 'Universitario'
            elif 'h.s.c' in educacion or 'a levels' in educacion:
                return 'Secundaria'
            elif 's.s.c' in educacion or 'o levels' in educacion:
                return 'Primaria/Secundaria'
            elif 'post graduate' in educacion:
                return 'Postgrado'
        return 'Otra/Desconocido'
    except:
        return 'Otra/Desconocido'

def categorizar_frecuencia(frecuencia):
    try:
        if isinstance(frecuencia, str):
            if 'never' in frecuencia or 'not applicable' in frecuencia:
                return 'Nunca'
            elif 'once/twice' in frecuencia:
                return '1-2 veces/semana'
            elif 'occasionally' in frecuencia:
                return 'Ocasionalmente'
            elif 'often' in frecuencia or 'regularly' in frecuencia:
                return 'Frecuentemente'
        return 'Desconocido'
    except:
        return 'Desconocido'

# Aplicar categorizaciones con manejo de errores
df['edad_categoria'] = df['Edad'].apply(categorizar_edad)
df['educacion_categoria'] = df['Educación'].apply(categorizar_educacion)
df['frecuencia_categoria'] = df['Frecuencia de uso de drogas'].apply(categorizar_frecuencia)

# Filtrar el dataset para solo incluir personas de 15-22 años
df_jovenes = df[df['edad_categoria'] == '15-22 años'].copy()

print(f"\nCategorías creadas:")
print(f"   - Edad: {df['edad_categoria'].unique()}")
print(f"\nFiltro aplicado - Solo personas de 15-22 años:")
print(f"   - Total registros originales: {len(df)}")
print(f"   - Registros 15-22 años: {len(df_jovenes)}")
print(f"   - Porcentaje del total: {(len(df_jovenes)/len(df)*100):.1f}%")

# Inicializar app Dash
app = Dash(__name__)

app.layout = html.Div([
    # Título principal
    html.H1(" Dashboard de Análisis de Adicción a Drogas en España - Jóvenes (15-22 años)", 
            style={
                'textAlign': 'center', 
                'color': '#2c3e50', 
                'marginBottom': '10px',
                'fontWeight': 'bold'
            }),
    
    html.P("Análisis Exploratorio de Datos - Factores de Riesgo y Perfiles de Consumo en Jóvenes",
           style={
               'textAlign': 'center',
               'color': '#7f8c8d',
               'fontSize': '16px',
               'marginBottom': '30px'
           }),
    
    # Información del filtro aplicado
    html.Div([
        html.Div([
            html.H4("🎯 Filtro Activo: Jóvenes de 15-22 años", 
                   style={'color': '#e74c3c', 'marginBottom': '5px', 'fontSize': '18px'}),
            html.P(f"Se muestran {len(df_jovenes)} registros de jóvenes en este rango de edad", 
                  style={'color': '#666', 'fontSize': '16px', 'margin': '0'}),
        ], style={
            'textAlign': 'center',
            'padding': '15px',
            'backgroundColor': '#fff5f5',
            'borderRadius': '10px',
            'border': '2px solid #e74c3c',
            'marginBottom': '20px'
        })
    ]),
    
    # Filtros (ahora solo para otros atributos, ya que edad está fijada)
    html.Div([
        html.H3("🔍 Filtros de Análisis Adicionales", 
                style={'color': '#34495e', 'marginBottom': '15px', 'fontSize': '20px'}),
        
        # Fila de filtros
        html.Div([
            # Filtro por Género
            html.Div([
                html.Label("Género:", 
                          style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='genero-filter',
                    options=[{'label': 'Todos', 'value': 'ALL'}] + 
                            [{'label': i.capitalize() if i != 'unknown' else 'Desconocido', 'value': i} 
                             for i in sorted(df_jovenes['Género'].unique())],
                    value='ALL',
                    style={'width': '100%', 'fontSize': '14px'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            # Filtro por Educación
            html.Div([
                html.Label("Nivel Educativo:", 
                          style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='educacion-filter',
                    options=[{'label': 'Todos', 'value': 'ALL'}] + 
                            [{'label': i, 'value': i} for i in sorted([x for x in df_jovenes['educacion_categoria'].unique() if x != 'unknown'])],
                    value='ALL',
                    style={'width': '100%', 'fontSize': '14px'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            # Filtro por Consumo de Drogas
            html.Div([
                html.Label("Ha consumido drogas:", 
                          style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='consumo-filter',
                    options=[{'label': 'Todos', 'value': 'ALL'},
                            {'label': 'Sí', 'value': 'yes'},
                            {'label': 'No', 'value': 'no'},
                            {'label': 'Desconocido', 'value': 'unknown'}],
                    value='ALL',
                    style={'width': '100%', 'fontSize': '14px'}
                ),
            ], style={'width': '30%', 'display': 'inline-block'}),
        ], style={'marginBottom': '20px'}),
        
        # Segunda fila de filtros
        html.Div([
            # Filtro por Frecuencia
            html.Div([
                html.Label("Frecuencia de consumo:", 
                          style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='frecuencia-filter',
                    options=[{'label': 'Todas', 'value': 'ALL'}] + 
                            [{'label': i, 'value': i} for i in sorted([x for x in df_jovenes['frecuencia_categoria'].unique() if x != 'unknown'])],
                    value='ALL',
                    style={'width': '100%', 'fontSize': '14px'}
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
            
            # Botón de reset
            html.Div([
                html.Label(" ", style={'display': 'block', 'marginBottom': '5px'}),
                html.Button('🔄 Reset Filtros', 
                           id='reset-button', 
                           n_clicks=0,
                           style={
                               'width': '100%',
                               'padding': '12px',
                               'backgroundColor': '#e74c3c',
                               'color': 'white',
                               'border': 'none',
                               'borderRadius': '5px',
                               'cursor': 'pointer',
                               'fontWeight': 'bold',
                               'fontSize': '16px'
                           })
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ]),
        
    ], style={
        'padding': '25px', 
        'backgroundColor': '#ecf0f1', 
        'borderRadius': '10px', 
        'marginBottom': '30px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # KPIs
    html.Div([
        html.H3("📊 Métricas Clave - Jóvenes 15-22 años", 
                style={'color': '#34495e', 'marginBottom': '20px', 'textAlign': 'center', 'fontSize': '22px'}),
        html.Div(id='kpi-cards', style={'textAlign': 'center'})
    ], style={'marginBottom': '30px'}),
    
    # Primera fila de gráficos - GRÁFICAS MÁS GRANDES
    html.Div([
        html.Div([
            dcc.Graph(id='pie-consumo')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='bar-genero')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ]),
    
    # Segunda fila de gráficos
    html.Div([
        html.Div([
            dcc.Graph(id='bar-educacion')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='bar-frecuencia')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ]),
    
    # Tercera fila de gráficos
    html.Div([
        html.Div([
            dcc.Graph(id='bar-problemas-mentales')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='bar-influencia-amigos')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ]),
    
    # Cuarta fila de gráficos - Heatmap más grande
    html.Div([
        html.Div([
            dcc.Graph(id='heatmap-correlacion')
        ], style={'width': '100%', 'display': 'inline-block', 'padding': '10px'}),
    ]),
    
    # Quinta fila - Gráfica comparativa más grande
    html.Div([
        html.Div([
            dcc.Graph(id='comparativa-edad')
        ], style={'width': '100%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    # Sexta fila - Probabilidad de consumo (modelo predictivo)
    html.Div([
        html.Div([
            dcc.Graph(id='probabilidad-consumo')
        ], style={'width': '100%', 'display': 'inline-block', 'padding': '10px'}),
    ]),
    
    # Estadísticas detalladas
    html.Div([
        html.H3("📋 Estadísticas Detalladas - Jóvenes 15-22 años", 
                style={'textAlign': 'center', 'marginTop': '30px', 'color': '#34495e', 'fontSize': '22px'}),
        html.Div(id='detailed-stats', style={'padding': '20px'})
    ]),

    # Footer
    html.Div([
        html.Hr(style={'margin': '40px 0'}),
        html.P("Dashboard de Análisis de Adicción a Drogas | Dataset: Drug Addiction in Spain | © 2026 | Filtro: Jóvenes 15-22 años",
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '14px'})
    ])
    
], style={
    'fontFamily': '"Segoe UI", Arial, sans-serif',
    'padding': '32px',
    'backgroundColor': '#f5f7fb',
    'maxWidth': '1800px',
    'margin': '24px auto',
    'borderRadius': '16px',
    'boxShadow': '0 8px 24px rgba(16, 24, 40, 0.08)',
    'lineHeight': '1.5'
})

# Callback para resetear filtros
@callback(
    [Output('genero-filter', 'value'),
     Output('educacion-filter', 'value'),
     Output('consumo-filter', 'value'),
     Output('frecuencia-filter', 'value')],
    Input('reset-button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return 'ALL', 'ALL', 'ALL', 'ALL'

# Función segura para crear gráficos
def create_empty_figure(title):
    fig = go.Figure()
    fig.add_annotation(
        text="No hay datos disponibles con los filtros actuales",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="gray")
    )
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', font=dict(size=20)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        width=800
    )
    return fig

# Función para crear gráficas con tamaño aumentado
def create_figure_with_increased_size(fig, title, height=500, width=800):
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', font=dict(size=22)),
        height=height,
        width=None,  # Se ajustará al contenedor
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
    )
    return fig

# Callback principal
@callback(
    [Output('kpi-cards', 'children'),
     Output('pie-consumo', 'figure'),
     Output('bar-genero', 'figure'),
     Output('bar-educacion', 'figure'),
     Output('bar-frecuencia', 'figure'),
     Output('bar-problemas-mentales', 'figure'),
     Output('bar-influencia-amigos', 'figure'),
     Output('heatmap-correlacion', 'figure'),
     Output('comparativa-edad', 'figure'),
    Output('probabilidad-consumo', 'figure'),
     Output('detailed-stats', 'children')],
    [Input('genero-filter', 'value'),
     Input('educacion-filter', 'value'),
     Input('consumo-filter', 'value'),
     Input('frecuencia-filter', 'value')]
)
def update_dashboard(genero_filter, educacion_filter, consumo_filter, frecuencia_filter):
    filtered_df = df_jovenes.copy()
    
    # Aplicar filtros con verificaciones
    try:
        if genero_filter != 'ALL':
            filtered_df = filtered_df[filtered_df['Género'] == genero_filter]
        
        if educacion_filter != 'ALL':
            filtered_df = filtered_df[filtered_df['educacion_categoria'] == educacion_filter]
        
        if consumo_filter != 'ALL':
            filtered_df = filtered_df[filtered_df['Alguna vez ha consumido drogas'] == consumo_filter]
        
        if frecuencia_filter != 'ALL':
            filtered_df = filtered_df[filtered_df['frecuencia_categoria'] == frecuencia_filter]
    except Exception as e:
        print(f"Error al aplicar filtros: {e}")
    
    # Calcular KPIs con manejo de errores
    try:
        total_personas = len(filtered_df)
        
        # Contar consumidores (que contengan 'yes' en la respuesta)
        consumidores = filtered_df['Alguna vez ha consumido drogas'].apply(
            lambda x: 1 if isinstance(x, str) and 'yes' in x.lower() else 0
        ).sum()
        
        porcentaje_consumo = (consumidores / total_personas * 100) if total_personas > 0 else 0
        
        # Grupo de edad más común (siempre será 15-22 años en este caso)
        promedio_edad = '15-22 años'
        
        # Problemas mentales (que no sean 'unknown')
        problemas_mentales = filtered_df['Problema mental/emocional'].apply(
            lambda x: 0 if isinstance(x, str) and ('unknown' in x.lower() or x.strip() == '') else 1
        ).sum()
        
        # Influencia de amigos (que contengan 'yes')
        con_influencia_amigos = filtered_df['Influencia de amigos'].apply(
            lambda x: 1 if isinstance(x, str) and 'yes' in x.lower() else 0
        ).sum()
        
        # Porcentaje con influencia de amigos
        porcentaje_influencia = (con_influencia_amigos / total_personas * 100) if total_personas > 0 else 0
        
        # Fumadores
        fumadores = filtered_df['Fumador'].apply(
            lambda x: 1 if isinstance(x, str) and 'yes' in x.lower() else 0
        ).sum()
        porcentaje_fumadores = (fumadores / total_personas * 100) if total_personas > 0 else 0
        
    except Exception as e:
        print(f"Error calculando KPIs: {e}")
        total_personas = 0
        consumidores = 0
        porcentaje_consumo = 0
        promedio_edad = '15-22 años'
        problemas_mentales = 0
        con_influencia_amigos = 0
        porcentaje_influencia = 0
        fumadores = 0
        porcentaje_fumadores = 0
    
    # Crear tarjetas de KPI con tamaño aumentado
    kpi_cards = html.Div([
        # KPI 1 - Total personas
        html.Div([
            html.H2(f"{total_personas:,}", 
                   style={'color': '#3498db', 'margin': '0', 'fontSize': '42px', 'fontWeight': 'bold'}),
            html.P("Jóvenes Analizados", 
                  style={'margin': '5px 0', 'color': '#666', 'fontSize': '16px'})
        ], style={
            'display': 'inline-block', 
            'width': '14%', 
            'textAlign': 'center', 
            'padding': '25px', 
            'backgroundColor': '#f8f9fa', 
            'borderRadius': '10px', 
            'margin': '5px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'border': '2px solid #3498db',
            'minHeight': '120px'
        }),
        
        # KPI 2 - Consumidores
        html.Div([
            html.H2(f"{consumidores:,}", 
                   style={'color': '#e74c3c', 'margin': '0', 'fontSize': '42px', 'fontWeight': 'bold'}),
            html.P("Han consumido drogas", 
                  style={'margin': '5px 0', 'color': '#666', 'fontSize': '16px'})
        ], style={
            'display': 'inline-block', 
            'width': '14%', 
            'textAlign': 'center',
            'padding': '25px', 
            'backgroundColor': '#f8f9fa', 
            'borderRadius': '10px', 
            'margin': '5px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'border': '2px solid #e74c3c',
            'minHeight': '120px'
        }),
        
        # KPI 3 - Porcentaje consumo
        html.Div([
            html.H2(f"{porcentaje_consumo:.1f}%", 
                   style={'color': '#2ecc71', 'margin': '0', 'fontSize': '42px', 'fontWeight': 'bold'}),
            html.P("Tasa de consumo", 
                  style={'margin': '5px 0', 'color': '#666', 'fontSize': '16px'})
        ], style={
            'display': 'inline-block', 
            'width': '14%', 
            'textAlign': 'center',
            'padding': '25px', 
            'backgroundColor': '#f8f9fa', 
            'borderRadius': '10px', 
            'margin': '5px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'border': '2px solid #2ecc71',
            'minHeight': '120px'
        }),
        
        # KPI 4 - Problemas mentales
        html.Div([
            html.H2(f"{problemas_mentales:,}", 
                   style={'color': '#f39c12', 'margin': '0', 'fontSize': '42px', 'fontWeight': 'bold'}),
            html.P("Problemas mentales", 
                  style={'margin': '5px 0', 'color': '#666', 'fontSize': '16px'})
        ], style={
            'display': 'inline-block', 
            'width': '14%', 
            'textAlign': 'center',
            'padding': '25px', 
            'backgroundColor': '#f8f9fa', 
            'borderRadius': '10px', 
            'margin': '5px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'border': '2px solid #f39c12',
            'minHeight': '120px'
        }),
        
        # KPI 5 - Influencia amigos
        html.Div([
            html.H2(f"{porcentaje_influencia:.1f}%", 
                   style={'color': '#1abc9c', 'margin': '0', 'fontSize': '42px', 'fontWeight': 'bold'}),
            html.P("Influencia de amigos", 
                  style={'margin': '5px 0', 'color': '#666', 'fontSize': '16px'})
        ], style={
            'display': 'inline-block', 
            'width': '14%', 
            'textAlign': 'center',
            'padding': '25px', 
            'backgroundColor': '#f8f9fa', 
            'borderRadius': '10px', 
            'margin': '5px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'border': '2px solid #1abc9c',
            'minHeight': '120px'
        }),
        
        # KPI 6 - Fumadores
        html.Div([
            html.H2(f"{porcentaje_fumadores:.1f}%", 
                   style={'color': '#9b59b6', 'margin': '0', 'fontSize': '42px', 'fontWeight': 'bold'}),
            html.P("Fumadores", 
                  style={'margin': '5px 0', 'color': '#666', 'fontSize': '16px'})
        ], style={
            'display': 'inline-block', 
            'width': '14%', 
            'textAlign': 'center',
            'padding': '25px', 
            'backgroundColor': '#f8f9fa', 
            'borderRadius': '10px', 
            'margin': '5px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'border': '2px solid #9b59b6',
            'minHeight': '120px'
        }),
    ], style={'textAlign': 'center'})
    
    # 1. Gráfico de pastel: Consumo de drogas - MÁS GRANDE
    try:
        if total_personas > 0:
            # Contar valores de consumo
            consumo_counts = filtered_df['Alguna vez ha consumido drogas'].value_counts()
            
            # Preparar etiquetas
            labels = []
            values = []
            
            for val, count in consumo_counts.items():
                if isinstance(val, str):
                    if 'yes' in val.lower():
                        labels.append('Sí')
                    elif 'no' in val.lower():
                        labels.append('No')
                    else:
                        labels.append('Desconocido')
                else:
                    labels.append('Desconocido')
                values.append(count)
            
            if values:
                fig_pie_consumo = px.pie(
                    values=values,
                    names=labels,
                    title='<b>Consumo de Drogas en Jóvenes (15-22 años)</b>',
                    color_discrete_sequence=['#e74c3c', '#2ecc71', '#95a5a6'],
                    hole=0.3
                )
                fig_pie_consumo.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Cantidad: %{value:,}<br>Porcentaje: %{percent}<extra></extra>',
                    textfont=dict(size=16)
                )
                fig_pie_consumo.update_layout(
                    height=550,
                    annotations=[dict(
                        text=f'Total: {total_personas} jóvenes',
                        x=0.5, y=0.5,
                        font_size=16,
                        showarrow=False,
                        font_color='#000000'
                    )],
                    legend=dict(
                        font=dict(size=14),
                        orientation="h",
                        yanchor="bottom",
                        y=-0.1,
                        xanchor="center",
                        x=0.5
                    )
                )
            else:
                fig_pie_consumo = create_empty_figure("Consumo de Drogas en Jóvenes")
        else:
            fig_pie_consumo = create_empty_figure("Consumo de Drogas en Jóvenes")
    except Exception as e:
        print(f"Error en pie chart: {e}")
        fig_pie_consumo = create_empty_figure("Consumo de Drogas en Jóvenes")
    
    # 2. Gráfico de barras: Distribución por género - MÁS GRANDE
    try:
        if total_personas > 0 and 'Género' in filtered_df.columns:
            genero_counts = filtered_df['Género'].value_counts()
            if len(genero_counts) > 0:
                fig_bar_genero = px.bar(
                    x=[g.capitalize() if g != 'unknown' else 'Desconocido' for g in genero_counts.index],
                    y=genero_counts.values,
                    title='<b>Distribución por Género (15-22 años)</b>',
                    labels={'x': 'Género', 'y': 'Cantidad'},
                    color=genero_counts.values,
                    color_continuous_scale='Reds'
                )
                fig_bar_genero.update_traces(
                    texttemplate='%{y:,}', 
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Cantidad: %{y:,}<extra></extra>',
                    textfont=dict(size=14)
                )
                fig_bar_genero.update_layout(
                    showlegend=False, 
                    coloraxis_showscale=False,
                    height=550,
                    xaxis=dict(tickfont=dict(size=14)),
                    yaxis=dict(tickfont=dict(size=14)),
                )
            else:
                fig_bar_genero = create_empty_figure("Distribución por Género")
        else:
            fig_bar_genero = create_empty_figure("Distribución por Género")
    except Exception as e:
        print(f"Error en bar chart género: {e}")
        fig_bar_genero = create_empty_figure("Distribución por Género")
    
    # 3. Gráfico de barras: Distribución por educación - MÁS GRANDE
    try:
        if total_personas > 0 and 'educacion_categoria' in filtered_df.columns:
            educacion_counts = filtered_df['educacion_categoria'].value_counts()
            if len(educacion_counts) > 0:
                fig_bar_educacion = px.bar(
                    x=educacion_counts.index,
                    y=educacion_counts.values,
                    title='<b>Nivel Educativo en Jóvenes (15-22 años)</b>',
                    labels={'x': 'Nivel Educativo', 'y': 'Cantidad'},
                    color=educacion_counts.values,
                    color_continuous_scale='Greens'
                )
                fig_bar_educacion.update_traces(
                    texttemplate='%{y:,}', 
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Cantidad: %{y:,}<extra></extra>',
                    textfont=dict(size=14)
                )
                fig_bar_educacion.update_layout(
                    showlegend=False, 
                    coloraxis_showscale=False,
                    height=550,
                    xaxis=dict(tickfont=dict(size=14)),
                    yaxis=dict(tickfont=dict(size=14)),
                )
            else:
                fig_bar_educacion = create_empty_figure("Distribución por Nivel Educativo")
        else:
            fig_bar_educacion = create_empty_figure("Distribución por Nivel Educativo")
    except Exception as e:
        print(f"Error en bar chart educación: {e}")
        fig_bar_educacion = create_empty_figure("Distribución por Nivel Educativo")
    
    # 4. Gráfico de barras: Frecuencia de consumo - MÁS GRANDE
    try:
        if total_personas > 0 and 'frecuencia_categoria' in filtered_df.columns:
            frecuencia_counts = filtered_df['frecuencia_categoria'].value_counts()
            if len(frecuencia_counts) > 0:
                fig_bar_frecuencia = px.bar(
                    x=frecuencia_counts.index,
                    y=frecuencia_counts.values,
                    title='<b>Frecuencia de Consumo en Jóvenes (15-22 años)</b>',
                    labels={'x': 'Frecuencia', 'y': 'Cantidad'},
                    color=frecuencia_counts.values,
                    color_continuous_scale='Purples'
                )
                fig_bar_frecuencia.update_traces(
                    texttemplate='%{y:,}', 
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Cantidad: %{y:,}<extra></extra>',
                    textfont=dict(size=14)
                )
                fig_bar_frecuencia.update_layout(
                    showlegend=False, 
                    coloraxis_showscale=False,
                    height=550,
                    xaxis=dict(tickfont=dict(size=14)),
                    yaxis=dict(tickfont=dict(size=14)),
                )
            else:
                fig_bar_frecuencia = create_empty_figure("Frecuencia de Consumo")
        else:
            fig_bar_frecuencia = create_empty_figure("Frecuencia de Consumo")
    except Exception as e:
        print(f"Error en bar chart frecuencia: {e}")
        fig_bar_frecuencia = create_empty_figure("Frecuencia de Consumo")
    
    # 5. Gráfico de barras: Problemas mentales/emocionales - MÁS GRANDE
    try:
        if total_personas > 0 and 'Problema mental/emocional' in filtered_df.columns:
            # Filtrar valores desconocidos y tomar top 10
            problemas_df = filtered_df[filtered_df['Problema mental/emocional'] != 'unknown']
            problemas_counts = problemas_df['Problema mental/emocional'].value_counts().head(10)
            
            if len(problemas_counts) > 0:
                fig_bar_problemas = px.bar(
                    x=problemas_counts.index,
                    y=problemas_counts.values,
                    title='<b>Problemas Mentales/Emocionales en Jóvenes</b>',
                    labels={'x': 'Problema', 'y': 'Cantidad'},
                    color=problemas_counts.values,
                    color_continuous_scale='Oranges'
                )
                fig_bar_problemas.update_traces(
                    texttemplate='%{y:,}', 
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Cantidad: %{y:,}<extra></extra>',
                    textfont=dict(size=12)
                )
                fig_bar_problemas.update_layout(
                    showlegend=False, 
                    coloraxis_showscale=False,
                    xaxis_tickangle=45,
                    height=550,
                    xaxis=dict(tickfont=dict(size=12)),
                    yaxis=dict(tickfont=dict(size=14)),
                )
            else:
                fig_bar_problemas = create_empty_figure("Problemas Mentales/Emocionales")
        else:
            fig_bar_problemas = create_empty_figure("Problemas Mentales/Emocionales")
    except Exception as e:
        print(f"Error en bar chart problemas: {e}")
        fig_bar_problemas = create_empty_figure("Problemas Mentales/Emocionales")
    
    # 6. Gráfico de barras: Influencia de amigos - MÁS GRANDE
    try:
        if total_personas > 0 and 'Influencia de amigos' in filtered_df.columns:
            influencia_counts = filtered_df['Influencia de amigos'].value_counts()
            if len(influencia_counts) > 0:
                # Preparar etiquetas más amigables
                labels = []
                values = []
                for val, count in influencia_counts.items():
                    if isinstance(val, str):
                        if 'yes' in val.lower():
                            labels.append('Sí influyen')
                        elif 'no' in val.lower():
                            labels.append('No influyen')
                        else:
                            labels.append('Desconocido')
                    else:
                        labels.append('Desconocido')
                    values.append(count)
                
                fig_bar_influencia = px.bar(
                    x=labels,
                    y=values,
                    title='<b>Influencia de Amigos en Jóvenes (15-22 años)</b>',
                    labels={'x': 'Influencia de amigos', 'y': 'Cantidad'},
                    color=values,
                    color_continuous_scale='Blues'
                )
                fig_bar_influencia.update_traces(
                    texttemplate='%{y:,}', 
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Cantidad: %{y:,}<extra></extra>',
                    textfont=dict(size=14)
                )
                fig_bar_influencia.update_layout(
                    showlegend=False, 
                    coloraxis_showscale=False,
                    height=550,
                    xaxis=dict(tickfont=dict(size=14)),
                    yaxis=dict(tickfont=dict(size=14)),
                )
            else:
                fig_bar_influencia = create_empty_figure("Influencia de Amigos")
        else:
            fig_bar_influencia = create_empty_figure("Influencia de Amigos")
    except Exception as e:
        print(f"Error en bar chart influencia: {e}")
        fig_bar_influencia = create_empty_figure("Influencia de Amigos")
    
    # 7. Heatmap de correlación - MÁS GRANDE
    try:
        if total_personas > 5:  # Necesitamos suficientes datos para correlación
            # Variables seleccionadas para correlación
            variables = [
                'Alguna vez ha consumido drogas',
                'Influencia de amigos',
                'Fumador',
                'Conflicto con la ley',
                'Vive con consumidor de drogas',
                'Problema mental/emocional'
            ]
            
            # Crear DataFrame para correlación
            corr_data = {}
            for var in variables:
                if var in filtered_df.columns:
                    # Convertir a valores binarios
                    corr_data[var] = filtered_df[var].apply(
                        lambda x: 1 if isinstance(x, str) and 'yes' in x.lower() else 0
                    )
            
            if corr_data and len(corr_data) > 1:
                corr_df = pd.DataFrame(corr_data)
                correlation_matrix = corr_df.corr()
                
                # Acortar nombres para mejor visualización
                short_names = {
                    'Alguna vez ha consumido drogas': 'Consumo drogas',
                    'Influencia de amigos': 'Influencia amigos',
                    'Fumador': 'Fumador',
                    'Conflicto con la ley': 'Conflicto ley',
                    'Vive con consumidor de drogas': 'Vive con consumidor',
                    'Problema mental/emocional': 'Problema mental'
                }
                
                correlation_matrix.index = [short_names.get(col, col) for col in correlation_matrix.index]
                correlation_matrix.columns = [short_names.get(col, col) for col in correlation_matrix.columns]
                
                fig_heatmap = px.imshow(
                    correlation_matrix,
                    title='<b>Matriz de Correlación entre Factores en Jóvenes</b>',
                    labels=dict(color="Correlación"),
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1,
                    text_auto='.2f',
                    aspect="auto"
                )
                fig_heatmap.update_layout(
                    height=600,
                    xaxis_title="Variables",
                    yaxis_title="Variables",
                    font=dict(size=14),
                    coloraxis_colorbar=dict(
                        title="Correlación",
                        thickness=20,
                        len=0.8
                    )
                )
                fig_heatmap.update_traces(
                    textfont=dict(size=12)
                )
            else:
                fig_heatmap = create_empty_figure("Matriz de Correlación")
        else:
            fig_heatmap = create_empty_figure("Matriz de Correlación")
    except Exception as e:
        print(f"Error en heatmap: {e}")
        fig_heatmap = create_empty_figure("Matriz de Correlación")
    
    # 8. Comparativa con otros grupos de edad - MÁS GRANDE
    try:
        # Calcular estadísticas por grupo de edad
        edad_stats = []
        
        for edad_cat in sorted(df['edad_categoria'].unique()):
            if edad_cat != 'Desconocido':
                df_edad = df[df['edad_categoria'] == edad_cat]
                
                # Porcentaje de consumo
                consumo_edad = df_edad['Alguna vez ha consumido drogas'].apply(
                    lambda x: 1 if isinstance(x, str) and 'yes' in x.lower() else 0
                ).sum()
                porcentaje_consumo_edad = (consumo_edad / len(df_edad) * 100) if len(df_edad) > 0 else 0
                
                edad_stats.append({
                    'Grupo de Edad': edad_cat,
                    'Porcentaje Consumo': porcentaje_consumo_edad,
                    'Total Personas': len(df_edad)
                })
        
        if edad_stats:
            edad_df = pd.DataFrame(edad_stats)
            
            # Crear gráfico de barras comparativas
            fig_comparativa = px.bar(
                edad_df,
                x='Grupo de Edad',
                y='Porcentaje Consumo',
                title='<b>Comparativa de Consumo por Grupo de Edad</b>',
                labels={'Porcentaje Consumo': 'Porcentaje que ha consumido (%)'},
                text='Porcentaje Consumo',
                color='Porcentaje Consumo',
                color_continuous_scale='Viridis'
            )
            
            # Destacar el grupo de 15-22 años
            colors = []
            for i, edad in enumerate(edad_df['Grupo de Edad']):
                if edad == '15-22 años':
                    colors.append('#e74c3c')  # Rojo para destacar
                else:
                    colors.append('#3498db')  # Azul para otros
            
            fig_comparativa.update_traces(
                marker_color=colors,
                texttemplate='%{y:.1f}%',
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Consumo: %{y:.1f}%<br>Total: %{customdata} personas<extra></extra>',
                customdata=edad_df['Total Personas'],
                textfont=dict(size=14)
            )
            
            fig_comparativa.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                height=550,
                yaxis_title="Porcentaje que ha consumido drogas (%)",
                xaxis=dict(tickfont=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
            )
            
            # Añadir anotación para destacar 15-22 años
            if len(edad_df) > 0:
                fig_comparativa.add_annotation(
                    x='15-22 años',
                    y=porcentaje_consumo + 5,
                    text=f"{porcentaje_consumo:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#e74c3c",
                    font=dict(size=14, color="#e74c3c", weight="bold")
                )
        else:
            fig_comparativa = create_empty_figure("Comparativa por Edad")
    except Exception as e:
        print(f"Error en comparativa por edad: {e}")
        fig_comparativa = create_empty_figure("Comparativa por Edad")

    # 9. Probabilidad de consumo (modelo predictivo) - Jóvenes 15-22
    # 9. Probabilidad de consumo (modelo predictivo) - Jóvenes 15-22
    try:
        if model_trained and feature_cols and total_personas > 0:
            X_pred = filtered_df[feature_cols].copy()
            probas = model_pipeline.predict_proba(X_pred)[:, 1]
            prob_mean = float(np.mean(probas)) if len(probas) > 0 else 0.0

            fig_prob = go.Figure()

            # Histograma
            fig_prob.add_trace(go.Histogram(
                x=probas,
                nbinsx=20,
                marker_color='#2ecc71',
                opacity=0.75,
                name='Distribución'
            ))

            # Línea vertical del promedio (CON ANOTACIÓN)
            fig_prob.add_vline(
                x=prob_mean,
                line_width=2,
                line_dash='dash',
                line_color='#e74c3c',
                annotation_text=f"Promedio: {prob_mean:.0%}",
                annotation_position="top right"
            )

            # Cajita de rangos de riesgo (SE AÑADE, NO REEMPLAZA)
            fig_prob.add_annotation(
                text=(
                    "<b>Rangos de riesgo</b><br>"
                    "🟢 Bajo: 0–33%<br>"
                    "🟡 Medio: 34–66%<br>"
                    "🔴 Alto: 67–100%"
                ),
                x=1.02, y=1.0,
                xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(44,62,80,0.35)",
                borderwidth=1,
                borderpad=8,
                font=dict(size=12, color="#2c3e50")
            )

            # Layout general
            fig_prob.update_layout(
                title='<b>Probabilidad Predicha de Inicio de Consumo (15-22 años)</b>',
                xaxis_title='Probabilidad de consumo (%)',
                yaxis_title='Cantidad de jóvenes',
                height=550,
                bargap=0.05,
                showlegend=False,
                xaxis=dict(range=[0, 1], tickformat='.0%', dtick=0.1),
                margin=dict(t=90, r=170)
            )

            # Bandas de riesgo (fondo)
            fig_prob.update_layout(
                shapes=[
                    dict(type='rect', xref='x', yref='paper',
                        x0=0, x1=0.33, y0=0, y1=1,
                        fillcolor='rgba(46, 204, 113, 0.08)', line=dict(width=0)),

                    dict(type='rect', xref='x', yref='paper',
                        x0=0.33, x1=0.66, y0=0, y1=1,
                        fillcolor='rgba(241, 196, 15, 0.08)', line=dict(width=0)),

                    dict(type='rect', xref='x', yref='paper',
                        x0=0.66, x1=1, y0=0, y1=1,
                        fillcolor='rgba(231, 76, 60, 0.08)', line=dict(width=0))
                ]
            )

        else:
            fig_prob = create_empty_figure("Probabilidad Predicha de Consumo")

    except Exception as e:
        print(f"Error en probabilidad de consumo: {e}")
        fig_prob = create_empty_figure("Probabilidad Predicha de Consumo")

    
    # Estadísticas detalladas - CON TEXTO MÁS GRANDE
    try:
        if total_personas > 0:
            # Estadísticas de consumo
            consumo_si = filtered_df['Alguna vez ha consumido drogas'].apply(
                lambda x: 1 if isinstance(x, str) and 'yes' in x.lower() else 0
            ).sum()
            consumo_no = filtered_df['Alguna vez ha consumido drogas'].apply(
                lambda x: 1 if isinstance(x, str) and 'no' in x.lower() else 0
            ).sum()
            consumo_unknown = total_personas - consumo_si - consumo_no
            
            # Distribución por género
            if 'Género' in filtered_df.columns:
                genero_stats = filtered_df['Género'].value_counts().head(3)
            else:
                genero_stats = pd.Series()
            
            # Top 3 problemas mentales
            if 'Problema mental/emocional' in filtered_df.columns:
                problemas_df = filtered_df[filtered_df['Problema mental/emocional'] != 'unknown']
                problemas_top = problemas_df['Problema mental/emocional'].value_counts().head(3)
            else:
                problemas_top = pd.Series()
            
            detailed_stats = html.Div([
                html.Div([
                    html.H4("Distribución de Consumo", style={'color': '#2c3e50', 'marginBottom': '15px', 'fontSize': '18px'}),
                    html.Table([
                        html.Tr([
                            html.Th("Estado", style={'padding': '12px', 'backgroundColor': '#e74c3c', 'color': 'white', 'textAlign': 'center', 'fontSize': '16px'}),
                            html.Th("Cantidad", style={'padding': '12px', 'backgroundColor': '#e74c3c', 'color': 'white', 'textAlign': 'center', 'fontSize': '16px'}),
                            html.Th("Porcentaje", style={'padding': '12px', 'backgroundColor': '#e74c3c', 'color': 'white', 'textAlign': 'center', 'fontSize': '16px'})
                        ]),
                        html.Tr([
                            html.Td("Ha consumido", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'}),
                            html.Td(f"{consumo_si:,}", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'}),
                            html.Td(f"{(consumo_si/total_personas*100):.1f}%" if total_personas > 0 else "0.0%", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'})
                        ]),
                        html.Tr([
                            html.Td("No ha consumido", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'}),
                            html.Td(f"{consumo_no:,}", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'}),
                            html.Td(f"{(consumo_no/total_personas*100):.1f}%" if total_personas > 0 else "0.0%", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'})
                        ]),
                        html.Tr([
                            html.Td("Desconocido", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'}),
                            html.Td(f"{consumo_unknown:,}", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'}),
                            html.Td(f"{(consumo_unknown/total_personas*100):.1f}%" if total_personas > 0 else "0.0%", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'})
                        ]),
                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': '#f8f9fa', 'margin': '0 auto'})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                
                html.Div([
                    html.H4("Distribución por Género", style={'color': '#2c3e50', 'marginBottom': '15px', 'fontSize': '18px'}),
                    html.Table([
                        html.Tr([
                            html.Th("Género", style={'padding': '12px', 'backgroundColor': '#3498db', 'color': 'white', 'textAlign': 'center', 'fontSize': '16px'}),
                            html.Th("Cantidad", style={'padding': '12px', 'backgroundColor': '#3498db', 'color': 'white', 'textAlign': 'center', 'fontSize': '16px'}),
                            html.Th("Porcentaje", style={'padding': '12px', 'backgroundColor': '#3498db', 'color': 'white', 'textAlign': 'center', 'fontSize': '16px'})
                        ]),
                    ] + (
                        [html.Tr([
                            html.Td(genero.capitalize() if genero != 'unknown' else 'Desconocido', style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'}),
                            html.Td(f"{count:,}", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'}),
                            html.Td(f"{(count/total_personas*100):.1f}%" if total_personas > 0 else "0.0%", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'})
                        ]) for genero, count in genero_stats.items()] if len(genero_stats) > 0 else [
                            html.Tr([
                                html.Td("No hay datos", colSpan=3, style={'padding': '10px', 'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '14px'})
                            ])
                        ]
                    ), style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': '#f8f9fa', 'margin': '0 auto'})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                
                html.Div([
                    html.H4("Top 3 Problemas Mentales", style={'color': '#2c3e50', 'marginBottom': '15px', 'fontSize': '18px'}),
                    html.Table([
                        html.Tr([
                            html.Th("Problema", style={'padding': '12px', 'backgroundColor': '#f39c12', 'color': 'white', 'textAlign': 'center', 'fontSize': '16px'}),
                            html.Th("Cantidad", style={'padding': '12px', 'backgroundColor': '#f39c12', 'color': 'white', 'textAlign': 'center', 'fontSize': '16px'}),
                            html.Th("Porcentaje", style={'padding': '12px', 'backgroundColor': '#f39c12', 'color': 'white', 'textAlign': 'center', 'fontSize': '16px'})
                        ]),
                    ] + (
                        [html.Tr([
                            html.Td(problema[:25] + "..." if len(problema) > 25 else problema, style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '13px'}),
                            html.Td(f"{count:,}", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'}),
                            html.Td(f"{(count/total_personas*100):.1f}%" if total_personas > 0 else "0.0%", style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '14px'})
                        ]) for problema, count in problemas_top.items()] if len(problemas_top) > 0 else [
                            html.Tr([
                                html.Td("No hay datos", colSpan=3, style={'padding': '10px', 'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '14px'})
                            ])
                        ]
                    ), style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': '#f8f9fa', 'margin': '0 auto'})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            ], style={'textAlign': 'center'})
        else:
            detailed_stats = html.Div([
                html.H4("No hay datos disponibles con los filtros actuales", 
                       style={'color': '#95a5a6', 'textAlign': 'center', 'padding': '20px', 'fontSize': '18px'})
            ])
    except Exception as e:
        print(f"Error en estadísticas detalladas: {e}")
        detailed_stats = html.Div([
            html.H4("Error al calcular estadísticas", 
                   style={'color': '#e74c3c', 'textAlign': 'center', 'padding': '20px', 'fontSize': '18px'})
        ])

    return (kpi_cards, fig_pie_consumo, fig_bar_genero, fig_bar_educacion, 
            fig_bar_frecuencia, fig_bar_problemas, fig_bar_influencia,
            fig_heatmap, fig_comparativa, fig_prob, detailed_stats)

if __name__ == '__main__':
    print("\nIniciando dashboard...")
    print("Abre tu navegador en: http://localhost:8052")
    app.run(debug=True, port=8052)
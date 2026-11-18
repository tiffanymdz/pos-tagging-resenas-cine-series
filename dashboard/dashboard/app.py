"""CineScope — Dashboard de análisis semántico. Ejecutar: python dashboard/app.py"""
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
RES = pd.read_csv(DATA / 'resenas.csv.gz', low_memory=False)
RES['id_resena'] = RES['id_resena'].astype(str)
for col in ['genero_principal','polaridad','origen_texto']:
    RES[col] = RES[col].fillna('Sin dato').astype(str)
RES['texto_es'] = RES['texto_es'].fillna('').astype(str)
TSNE = pd.read_csv(DATA / 'tsne.csv.gz')
POLI = pd.read_csv(DATA / 'polisemia.csv')
MASK = pd.read_csv(DATA / 'masked_lm.csv')
MASK_CTX = pd.read_csv(DATA / 'masked_lm_contexto.csv')
BUSQ = pd.read_csv(DATA / 'busquedas.csv')
for col in ['id_resena']:
    TSNE[col] = TSNE[col].astype(str)
    BUSQ[col] = BUSQ[col].astype(str)

METRIC_PATH = DATA / 'resumen_comparacion.csv'
METRICS = pd.read_csv(METRIC_PATH) if METRIC_PATH.exists() else None
COLORS = {'bg':'#0a0b16','surface':'#141729','text':'#f2f2fb','muted':'#a2a6bd','purple':'#a58bff','pink':'#ff6ea8','cyan':'#56d6dc'}
PAL = ['#a58bff','#ff6ea8','#56d6dc','#f8bb6b','#7ce3a4','#e4a2ff','#8aa9ff','#fa8b86']

app = Dash(__name__, title='CineScope | Semantic Intelligence', suppress_callback_exceptions=True)
server = app.server

def chart(fig, height=370):
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'family':'Inter, Segoe UI, Arial','color':'#e9e9f5'},
        margin=dict(l=22,r=18,t=48,b=28), height=height,
        legend=dict(orientation='h',y=-0.20),
        hoverlabel=dict(bgcolor='#22253a',font_size=12),
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,.065)',zeroline=False)
    fig.update_yaxes(gridcolor='rgba(255,255,255,.065)',zeroline=False)
    return fig

def panel(children, title=None, subtitle=None, className=''):
    head = [html.H3(title),html.P(subtitle,className='muted')] if title else []
    return html.Div(head+list(children),className='panel '+className)

def stat(label, value, note, symbol):
    return html.Div([html.Div([html.Span(symbol,className='stat-icon'),html.Span(label,className='stat-label')],className='stat-top'),html.Div(value,className='stat-number'),html.Div(note,className='muted stat-note')],className='stat')

def pill(text):return html.Span(text,className='pill')

app.layout = html.Div([
    html.Div([
        html.Div([html.Div('✦',className='logo-symbol'),html.Div([html.Strong('CineScope'),html.Small('SEMANTIC INTELLIGENCE')])],className='brand'),
        html.Div('LABORATORIO DE LENGUAJE',className='side-caption'),
        html.A('◈  Panorama general',href='#panorama',className='nav-item'),
        html.A('◎  Explorador de reseñas',href='#explorador',className='nav-item'),
        html.A('✧  Espacio de embeddings',href='#embeddings',className='nav-item'),
        html.A('▣  Comparación de modelos',href='#comparacion',className='nav-item'),
        html.A('⌕  Búsqueda semántica',href='#busqueda',className='nav-item'),
        html.A('◌  Polisemia & Masked LM',href='#lenguaje',className='nav-item'),
        html.Div([html.Div('PROYECTO 02',className='footer-title'),html.Div('Análisis semántico de reseñas de cine y series'),html.Small('Plotly Dash · BETO · Word2Vec')],className='side-footer'),
    ],className='sidebar'),
    html.Main([
        html.Div([html.Span('PROYECTO ACADÉMICO  /  NLP',className='eyebrow'),pill('●  DATOS REALES')],className='topbar'),
        html.Section([html.Div('ENTENDÉ LO QUE DICEN LAS RESEÑAS',className='hero-eyebrow'),html.H1(['El cine, visto a través ',html.Span('del lenguaje.')]),html.P('Explorá emociones, géneros y patrones semánticos en un corpus de reseñas en español. De la frecuencia de palabras a los embeddings contextuales.'),html.Div([pill('BoW + TF-IDF'),pill('Word2Vec'),pill('BETO')],className='hero-pills')],className='hero',id='panorama'),
        html.Div([
            stat('RESEÑAS ANALIZADAS',f'{len(RES):,}'.replace(',','.'),'Corpus consolidado','◈'),
            stat('GÉNEROS',str(RES.loc[RES.genero_principal!='Sin dato','genero_principal'].nunique()),'Categorías en el corpus','▦'),
            stat('EMBEDDINGS BETO','768','Dimensiones por reseña','✧'),
            stat('ORÍGENES',str(RES.loc[RES.origen_texto!='Sin dato','origen_texto'].nunique()),'Fuentes de texto','◎'),
        ],className='stat-grid'),
        html.Section([
            html.Div([html.H2('Panorama del corpus'),html.P('Distribución real de géneros, valoraciones y procedencia.',className='muted')],className='section-heading'),
            html.Div([panel([dcc.Graph(id='genre-graph',config={'displayModeBar':False})],title='Géneros principales',className='wide'),panel([dcc.Graph(id='polar-graph',config={'displayModeBar':False})],title='Polaridad')],className='two-col'),
            html.Div([panel([dcc.Graph(id='origin-graph',config={'displayModeBar':False})],title='Origen de las reseñas'),panel([html.Div([html.Div('¿QUÉ ESTAMOS ANALIZANDO?',className='eyebrow'),html.H3('De palabras a significado'),html.P('BoW y TF-IDF cuantifican términos; Word2Vec aprende relaciones estáticas entre palabras; BETO produce representaciones sensibles al contexto.'),html.P('Las visualizaciones t-SNE son exploratorias y no equivalen a una medida de calidad del modelo.',className='muted')],className='explainer')],title='Acerca del proyecto')],className='two-col'),
        ],className='section'),
        html.Section([
            html.Div([html.H2('Explorador de reseñas'),html.P('Filtrá el corpus y leé las reseñas originales.',className='muted')],className='section-heading',id='explorador'),
            panel([html.Div([html.Div([html.Label('GÉNERO'),dcc.Dropdown(id='genre-filter',options=[{'label':'Todos','value':'__all__'}]+[{'label':x,'value':x} for x in sorted(RES.genero_principal.unique())],value='__all__',clearable=False)],className='filter'),html.Div([html.Label('POLARIDAD'),dcc.Dropdown(id='polar-filter',options=[{'label':'Todas','value':'__all__'}]+[{'label':x,'value':x} for x in sorted(RES.polaridad.unique())],value='__all__',clearable=False)],className='filter'),html.Div([html.Label('ORIGEN'),dcc.Dropdown(id='origin-filter',options=[{'label':'Todos','value':'__all__'}]+[{'label':x,'value':x} for x in sorted(RES.origen_texto.unique())],value='__all__',clearable=False)],className='filter'),html.Div([html.Label('BUSCAR TEXTO'),dcc.Input(id='text-filter',placeholder='Palabra o frase...',debounce=True)],className='filter')],className='filters'),html.Div(id='review-count',className='muted'),html.Div(id='review-cards',className='review-grid'),html.Div([html.Button('Anterior',id='prev-page',n_clicks=0),html.Span(id='page-label'),html.Button('Siguiente',id='next-page',n_clicks=0)],className='pagination')]),
        ],className='section'),
        html.Section([
            html.Div([html.H2('Espacio de embeddings'),html.P('Proyecciones t-SNE 2D sobre la misma muestra de reseñas (550 observaciones).',className='muted')],className='section-heading',id='embeddings'),
            panel([html.Div([html.Div([html.Label('REPRESENTACIÓN'),dcc.Dropdown(id='model-select',options=[{'label':x,'value':x} for x in ['BETO','CBOW','Skip-Gram','BoW','TF-IDF']],value='BETO',clearable=False)],className='filter'),html.Div([html.Label('COLOREAR POR'),dcc.RadioItems(id='color-select',options=[{'label':'Género','value':'genero_principal'},{'label':'Polaridad','value':'polaridad'},{'label':'Origen','value':'origen_texto'}],value='polaridad',inline=True)],className='filter')],className='filters'),dcc.Graph(id='tsne-graph',config={'displayModeBar':False}),html.P('t-SNE calculado previamente con semilla fija. Los ejes y las distancias entre gráficos de distintos modelos no son directamente comparables.',className='muted footnote')]),
        ],className='section'),
        html.Section([
            html.Div([html.H2('Comparación de representaciones'),html.P('Resultados cuantitativos del notebook 05: clasificación y clustering.',className='muted')],className='section-heading',id='comparacion'),
            html.Div(id='metrics-area'),
        ],className='section'),
        html.Section([
            html.Div([html.H2('Búsqueda semántica con BETO'),html.P('Consultas y resultados calculados previamente con el corpus completo.',className='muted')],className='section-heading',id='busqueda'),
            panel([html.Label('ELEGÍ UNA CONSULTA EVALUADA'),dcc.Dropdown(id='query-select',options=[{'label':x,'value':x} for x in BUSQ['consulta'].dropna().unique()],value=BUSQ['consulta'].iloc[0],clearable=False),html.Div(id='query-results',className='results-list'),html.P('Resultados precalculados: para realizar búsquedas nuevas en vivo se necesita desplegar también el modelo BETO.',className='muted footnote')]),
        ],className='section'),
        html.Section([
            html.Div([html.H2('Laboratorio de lenguaje'),html.P('Polisemia contextual y predicción de palabras ocultas con BETO.',className='muted')],className='section-heading',id='lenguaje'),
            html.Div([panel([dcc.Graph(id='poly-graph',config={'displayModeBar':False}),html.Div(id='poly-examples')],title='Polisemia: similitud coseno'),panel([html.Label('ELEGÍ UNA FRASE'),dcc.Dropdown(id='mask-select',options=[{'label':x[:90],'value':x} for x in pd.concat([MASK,MASK_CTX])['frase'].dropna().unique()],value=MASK['frase'].iloc[0],clearable=False),dcc.Graph(id='mask-graph',config={'displayModeBar':False})],title='Masked Language Model')],className='two-col'),
        ],className='section'),
        html.Footer('CINESCOPE  /  ANÁLISIS SEMÁNTICO DE RESEÑAS  ·  DASHBOARD ACADÉMICO')
    ],className='main')
],className='shell')

@app.callback(Output('genre-graph','figure'),Output('polar-graph','figure'),Output('origin-graph','figure'),Input('genre-filter','value'))
def overview(_):
    counts=RES.genero_principal.value_counts().head(12).sort_values()
    a=px.bar(x=counts.values,y=counts.index,orientation='h',color_discrete_sequence=[COLORS['purple']]);a.update_layout(xaxis_title='',yaxis_title='',showlegend=False)
    pc=RES.polaridad.value_counts();b=px.pie(names=pc.index,values=pc.values,hole=.72,color_discrete_sequence=PAL);b.update_traces(textinfo='percent+label',textfont_size=12)
    oc=RES.origen_texto.value_counts();c=px.bar(x=oc.index,y=oc.values,color_discrete_sequence=[COLORS['cyan']]);c.update_layout(xaxis_title='',yaxis_title='Reseñas')
    return chart(a),chart(b),chart(c,300)

@app.callback(Output('review-count','children'),Output('review-cards','children'),Output('page-label','children'),Input('genre-filter','value'),Input('polar-filter','value'),Input('origin-filter','value'),Input('text-filter','value'),Input('prev-page','n_clicks'),Input('next-page','n_clicks'))
def reviews(genre,polar,origin,search,prev,nxt):
    d=RES
    if genre!='__all__':d=d[d.genero_principal==genre]
    if polar!='__all__':d=d[d.polaridad==polar]
    if origin!='__all__':d=d[d.origen_texto==origin]
    if search:d=d[d.texto_es.str.contains(str(search),case=False,regex=False,na=False)]
    page=max(0,(nxt or 0)-(prev or 0)); pages=max(1,int(np.ceil(len(d)/6)));page=min(page,pages-1)
    items=[]
    for _,r in d.iloc[page*6:page*6+6].iterrows():
        txt=r.texto_es[:420]+('…' if len(r.texto_es)>420 else '')
        items.append(html.Div([html.Div([pill(r.genero_principal),pill(r.polaridad)],className='tag-row'),html.P(txt),html.Small(f"ID: {r.id_resena}  ·  {r.origen_texto}")],className='review-card'))
    if not items:items=[html.P('No se encontraron reseñas con esos filtros.',className='muted')]
    return f'{len(d):,} reseñas encontradas'.replace(',','.'),items,f'Página {page+1} / {pages}'

@app.callback(Output('tsne-graph','figure'),Input('model-select','value'),Input('color-select','value'))
def tsne(model,col):
    d=TSNE[TSNE.modelo==model]
    f=px.scatter(d,x='x',y='y',color=col,hover_data=['id_resena','genero_principal','polaridad','origen_texto'],color_discrete_sequence=PAL,opacity=.78)
    f.update_traces(marker=dict(size=8,line=dict(width=.3,color='#101124')))
    f.update_layout(xaxis_title='t-SNE 1',yaxis_title='t-SNE 2',legend_title_text='')
    return chart(f,520)

@app.callback(Output('metrics-area','children'),Input('model-select','value'))
def metrics(_):
    if METRICS is None:
        return panel([html.Div('◷',className='empty-icon'),html.H3('Esperando métricas de comparación'),html.P('Ejecutá el notebook 05_comparacion_final.ipynb y copiá resumen_comparacion.csv dentro de dashboard/data/. Después reiniciá la aplicación.',className='muted'),html.P('No se muestran métricas inventadas: esta sección se activará con los resultados reales del equipo.',className='muted')],className='empty-state')
    d=METRICS.copy()
    d['representacion']=d['representacion'].astype(str)
    labels={'accuracy':'Exactitud (Accuracy)','f1_macro':'F1 macro','f1_weighted':'F1 ponderado','silhouette_coseno':'Silhouette (coseno)','ARI_vs_polaridad':'ARI frente a polaridad'}
    best_acc=d.loc[d.accuracy.idxmax()]
    best_f1=d.loc[d.f1_macro.idxmax()]
    best_sil=d.loc[d.silhouette_coseno.idxmax()]
    highlights=html.Div([
        stat('MEJOR EXACTITUD',f"{best_acc.accuracy:.1%}",best_acc.representacion,'◈'),
        stat('MEJOR F1 MACRO',f"{best_f1.f1_macro:.3f}",best_f1.representacion,'✧'),
        stat('MEJOR SILHOUETTE',f"{best_sil.silhouette_coseno:.3f}",best_sil.representacion,'◎'),
    ],className='stat-grid')
    charts=[]
    for col in ['accuracy','f1_macro','silhouette_coseno','ARI_vs_polaridad']:
        dd=d.sort_values(col,ascending=True)
        fig=px.bar(dd,x=col,y='representacion',orientation='h',color='representacion',
                   color_discrete_sequence=PAL,text=col)
        fig.update_traces(texttemplate='%{x:.3f}',textposition='outside',cliponaxis=False)
        fig.update_layout(showlegend=False,xaxis_title=labels[col],yaxis_title='',
                          margin=dict(l=25,r=60,t=40,b=35))
        charts.append(panel([dcc.Graph(figure=chart(fig,355),config={'displayModeBar':False})],title=labels[col]))
    note=panel([
        html.P('BoW presenta la mayor exactitud; TF-IDF lidera F1 macro; CBOW logra el mayor Silhouette coseno. Son métricas distintas: no indican un ganador universal.'),
        html.P('Los valores Silhouette son bajos y los ARI están cerca de cero (algunos son negativos): los grupos de K-Means no reproducen claramente las etiquetas de polaridad.',className='muted'),
        html.P('Comparación basada en el notebook 05; el clustering utiliza k=3. Las proyecciones t-SNE son ilustrativas y no equivalen a una métrica de calidad.',className='muted')
    ],title='Cómo interpretar los resultados')
    table=panel([dash_table.DataTable(
        data=d.round(4).to_dict('records'),
        columns=[{'name':labels.get(c,c.replace('_',' ').title()),'id':c} for c in d.columns],
        style_table={'overflowX':'auto'},
        style_header={'backgroundColor':'#20243a','color':'white','fontWeight':'bold'},
        style_cell={'backgroundColor':'#15182a','color':'#f2f2fb','padding':'12px','border':'1px solid #2b2f48','textAlign':'left'}
    )],title='Resultados completos')
    return html.Div([highlights,html.Div(charts,className='metric-grid'),note,table])

@app.callback(Output('query-results','children'),Input('query-select','value'))
def search(q):
    d=BUSQ[BUSQ.consulta==q].sort_values('similitud',ascending=False)
    return [html.Div([html.Div([html.Span(f'#{i+1}',className='rank'),html.Span(f'{float(r.similitud):.4f}',className='score'),pill(str(r.genero_principal)),pill(str(r.polaridad))],className='result-head'),html.P(str(r.texto_es)),html.Small('Reseña '+str(r.id_resena))],className='result-item') for i,(_,r) in enumerate(d.iterrows())]

@app.callback(Output('poly-graph','figure'),Output('poly-examples','children'),Input('mask-select','value'))
def poly(_):
    d=POLI.sort_values('similitud_coseno');f=px.bar(d,x='similitud_coseno',y='palabra',orientation='h',color='similitud_coseno',color_continuous_scale=['#ff6ea8','#a58bff','#56d6dc'],range_color=[0,1]);f.update_layout(coloraxis_showscale=False,xaxis_range=[0,1],xaxis_title='Similitud coseno',yaxis_title='')
    return chart(f,330),html.P('Menor similitud indica representaciones más diferentes entre contextos; no demuestra por sí sola comprensión correcta del significado.',className='muted footnote')

@app.callback(Output('mask-graph','figure'),Input('mask-select','value'))
def mask(q):
    d=pd.concat([MASK,MASK_CTX],ignore_index=True);d=d[d.frase==q].sort_values('probabilidad')
    f=px.bar(d,x='probabilidad',y='prediccion',orientation='h',color_discrete_sequence=[COLORS['pink']]);f.update_layout(xaxis_title='Probabilidad del token',yaxis_title='',xaxis_tickformat='.0%')
    return chart(f,330)

if __name__ == '__main__':
    app.run(debug=False,host='127.0.0.1',port=8050)

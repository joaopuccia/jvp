import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import chardet

# Helper functions
@st.cache_data
def safe_read_excel(uploaded_file) -> pd.DataFrame:
    """Read a excel file with multiple encoding attempts."""
    if uploaded_file is None:
        return None
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']
    for enc in encodings:
        try:
            df = pd.read_excel(uploaded_file, index_col = 0)
            df.index = pd.to_datetime(df.index, errors='coerce')
            return df
        except Exception:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            continue
    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        detection = chardet.detect(raw)
        encoding = detection.get('encoding', 'utf-8')
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f'Erro ao ler o arquivo: {e}')
        return None

# @st.cache_data
# def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # """Convert index to datetime and all columns to numeric where possible."""
    # if df is None:
        # return None
    # Convert index
    # try:
        # if not isinstance(df.index, pd.DatetimeIndex):
            # df.index = pd.to_datetime(df.index, errors='coerce')
        # if df.index.isna().all():
            # for col in df.columns:
                # col_dt = pd.to_datetime(df[col], errors='coerce')
                # if col_dt.notna().sum() > 0:
                    # df.index = col_dt
                    # df = df.drop(columns=[col])
                    # break
    # except Exception:
        # pass
    # Convert columns to numeric
    # for col in df.columns:
        # df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop all NaN rows
    # df = df.dropna(how='all')
    # df = df.dropna()
    # return df

@st.cache_data
def run_arimax(series: pd.Series, order=(10,1,1), lag=7, horizon=10):
    """Run ARIMAX model where exog is series shifted by lag; predict for horizon steps."""
    series = series.sort_index()
    exog = series.shift(lag)
    mask = (~exog.isna()) & (~series.isna())
    series_train = series[mask]
    exog_train = exog[mask]
    try:
        model = SARIMAX(series_train, exog=exog_train, order=order, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
    except Exception as e:
        st.error(f'Erro ao ajustar o modelo ARIMAX: {e}')
        return None, None
    last_exog = exog_train.iloc[-1] if len(exog_train) > 0 else 0
    exog_forecast = np.repeat(last_exog, horizon)
    try:
        pred = results.get_forecast(steps=horizon, exog=exog_forecast)
        pred_mean = pred.predicted_mean
    except Exception:
        pred_mean = pd.Series([np.nan] * horizon)
    return pred_mean, results

# Streamlit interface
st.set_page_config(page_title='Dashboard de Previsão de Preço de Energia', layout='wide')
st.title('Dashboard de Previsão de Preço de Energia')

# Sidebar
st.sidebar.header('Controles do Modelo')
pred_horizon = st.sidebar.slider('Horizonte de Previsão (dias)', min_value=1, max_value=30, value=10)
lag = st.sidebar.slider('Lag do Modelo (períodos)', min_value=0, max_value=20, value=7)
order = (st.sidebar.number_input('AR order p', min_value=0, max_value=20, value=10),
         st.sidebar.number_input('Integration order d', min_value=0, max_value=3, value=1),
         st.sidebar.number_input('MA order q', min_value=0, max_value=20, value=1))
uploaded_file = st.sidebar.file_uploader('Carregar excel de dados', type=['xlsx'])

# Load and preprocess
if uploaded_file is not None:
    df_raw = safe_read_excel(uploaded_file)
    df = df_raw
    # df = preprocess_df(df_raw)
else:
    # Simulate some data if no file
    periods = 100
    dates = pd.date_range(end=pd.Timestamp.today(), periods=periods)
    np.random.seed(42)
    price = np.cumsum(np.random.normal(loc=0.0, scale=1.0, size=periods)) + 50 + 5 * np.sin(np.linspace(0, np.pi*4, periods))
    ena    = np.cumsum(np.random.normal(size=periods)) + 30
    hidro  = np.cumsum(np.random.normal(size=periods)) + 20
    termica= np.cumsum(np.random.normal(size=periods)) + 10
    eolica = np.cumsum(np.random.normal(size=periods)) + 15
    cambio = 5 + np.random.normal(scale=0.5, size=periods)
    df = pd.DataFrame({'Price': price, 'ENA': ena, 'Hidrica': hidro, 'Termica': termica, 'Eolica': eolica, 'Cambio': cambio}, index=dates)

# If df exists and ready
if df is not None and 'Price' in df.columns:
    # Run ARIMAX
    pred, results = run_arimax(df['Price'], order=order, lag=lag, horizon=pred_horizon)
    # KPIs
    aic = round(results.aic,3) if results is not None else None
    llf = round(results.llf,3) if results is not None else None
    # Tests
    if results is not None:
        jb_p = round(jarque_bera(results.resid)[1], 3)
        lj = acorr_ljungbox(results.resid, lags=[lag], return_df=True)
        lb_p = round(lj['lb_pvalue'].iloc[-1], 3)
        arch = het_arch(results.resid)
        arch_p = round(arch[3], 3)
    else:
         jb_p, lb_p, arch_p = (None, None, None)
        
        
    # Prediction DataFrame
    
    last_date = pd.to_datetime(df.index[-1])
    pred_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_horizon, freq='W')
    pred_df = pd.DataFrame({'Price': pred.values}, index=pred_index)

    # Main Panel
    st.subheader('Preço Histórico e Previsão')
    fig = px.line(x=df.index, y=df['Price'], labels={'x': 'Data', 'y': 'Preço (R$)'}, title='Histórico de Preços')
    if pred is not None:
        fig.add_scatter(x=pred_df.index, y=pred_df['Price'], mode='lines+markers', name='Previsão', line=dict(dash='dash', color='orange'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('Próximos valores previstos:')
    pred_table = pd.DataFrame({'Data': pred_df.index.to_series().dt.strftime('%Y-%m-%d'), 'Preço Previsto': np.round(pred_df['Price'].values, 2)})
    last_val = df['Price'].values[-1]
    pred_table['Variação vs Último'] = np.round((pred_df['Price'].values - last_val) / last_val * 100, 2)
    pred_table = pred_table.set_index('Data')
    st.table(pred_table)

    st.subheader('Alerta de Momento de Contratação')
    # if pred is not None and (pred_df['Price'].mean() < df['Price'].mean()):
        # st.success('Bom momento para contratar: preço previsto em queda em relação à média histórica')
    # else:
        # st.error('Não é um bom momento para contratar: preço previsto acima da média histórica')
        
        
if pred is not None and len(pred_df) >= 2:
    preco_inicial = pred_df['Price'].iloc[0]
    preco_final = pred_df['Price'].iloc[-1]
    variacao_pct = ((preco_final - preco_inicial) / preco_inicial) * 100

    # Tolerância de ±2%
    if variacao_pct > 2:
        simbolo = "⬆️"
        mensagem = f"Tendência de alta {simbolo} (+{variacao_pct:.2f}%). Bom momento para contratar."
        st.success(mensagem)
    elif variacao_pct < -2:
        simbolo = "⬇️"
        mensagem = f"Tendência de queda {simbolo} ({variacao_pct:.2f}%). Mau momento para contratar."
        st.error(mensagem)
    else:
        simbolo = "➖"
        mensagem = f"Manutenção dos preços {simbolo} (variação de {variacao_pct:.2f}%). Tendência estável."
        st.info(mensagem)

    st.subheader('Indicadores de Qualidade do Modelo')
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('AIC', aic)
    c2.metric('Log-Likelihood', llf)
    c3.metric('p-value Ljung-Box', lb_p)
    c4.metric('p-value Jarque-Bera', jb_p)
    c5.metric('p-value Heteroscedasticidade', arch_p)

    # =========================
    # QUADRO DE QUALIDADE (0–100)
    # =========================
    st.subheader('Indicadores de Qualidade do Modelo2')
    # Funções auxiliares de pontuação
    def pvalue_points(p):
        """Converte p-value em 0–20, premiando p>=0.05 (nível 5%)."""
        if p is None or np.isnan(p):
            return 0
        # Escala linear até 0.05; acima disso, teto de 20
        return float(np.clip(20.0 * (p / 0.05), 0.0, 20.0))

    def aic_points(aic, n):
        """Pontuação 0–20 para AIC normalizado por observação (menor é melhor)."""
        if aic is None or n is None or n <= 0:
            return 0
        aic_per = aic / n
        if aic_per <= 5:
            pts, status = 20, "Excelente"
        elif aic_per <= 10:
            pts, status = 15, "Bom"
        elif aic_per <= 15:
            pts, status = 10, "Regular"
        else:
            pts, status = 0, "Ruim"
        return pts, status, aic_per

    def llf_points(llf, n):
        """Pontuação 0–20 para LLF normalizado por observação (maior é melhor)."""
        if llf is None or n is None:
            return 0, "Ruim", np.nan
        llf_per = llf / n
        if llf_per >= -0.5:
            pts, status = 20, "Excelente"
        elif llf_per >= -1:
            pts, status = 15, "Bom"
        elif llf_per >= -2:
            pts, status = 10, "Regular"
        else:
            pts, status = 0, "Ruim"
        return pts, status, llf_per

    # KPIs e testes
    if results is not None:
        aic = results.aic
        llf = results.llf
        resid = results.resid
        n_obs = len(resid) if resid is not None else None

        # jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
        # lj = acorr_ljungbox(resid, lags=[lag], return_df=True)
        # lb_p = float(lj['lb_pvalue'].iloc[-1]) if lj is not None else np.nan
        # arch = het_arch(resid)
        # arch_p = float(arch[3]) if arch is not None else np.nan

        # Pontos por critério (cada um máximo 20)
        jb_pts = pvalue_points(jb_p)
        lb_pts = pvalue_points(lb_p)
        arch_pts = pvalue_points(arch_p)
        aic_pts, aic_status, aic_per = aic_points(aic, n_obs)
        llf_pts, llf_status, llf_per = llf_points(llf, n_obs)

        total_score = int(round(jb_pts + lb_pts + arch_pts + aic_pts + llf_pts))
        if total_score >= 85:
            qualidade = "Excelente"
            msg_funil = "Modelo muito bem calibrado; resíduos comportados e ajuste consistente."
            alert_fn = st.success
        elif total_score >= 70:
            qualidade = "Bom"
            msg_funil = "Modelo adequado; pequenas melhorias podem elevar a robustez."
            alert_fn = st.success
        elif total_score >= 50:
            qualidade = "Regular"
            msg_funil = "Reveja ordem (p,d,q), transformação, ou exógenas para reduzir AIC e melhorar resíduos."
            alert_fn = st.warning
        else:
            qualidade = "Ruim"
            msg_funil = "Modelo precisa de revisão: autocorrelação/heterocedasticidade ou ajuste fraco."
            alert_fn = st.error

        # Monta o quadro
        rows = [
            ["Normalidade (Jarque–Bera)", f"p={jb_p:.3f}", "OK" if jb_p >= 0.05 else "Falhou", round(jb_pts, 1)],
            ["Autocorrelação (Ljung–Box, lag 10)", f"p={lb_p:.3f}", "OK" if lb_p >= 0.05 else "Falhou", round(lb_pts, 1)],
            ["Heterocedasticidade (ARCH)", f"p={arch_p:.3f}", "OK" if arch_p >= 0.05 else "Falhou", round(arch_pts, 1)],
            ["AIC por observação", f"{aic_per:.3f}", aic_status, aic_pts],
            ["LLF por observação", f"{llf_per:.3f}", llf_status, llf_pts],
        ]
        quadro = pd.DataFrame(rows, columns=["Critério", "Métrica", "Status", "Pontos"])

    else:
        total_score, qualidade, msg_funil = 0, "Ruim", "Modelo não ajustado."
        quadro = pd.DataFrame(columns=["Critério", "Métrica", "Status", "Pontos"])

    st.dataframe(quadro.set_index('Critério'), use_container_width=True)

    st.subheader('Quadro de Qualidade da Previsão (ARIMAX)')
    st.metric('Score (0–100)', total_score)
    alert_fn(f'Qualidade: {qualidade}. {msg_funil}')

    # Gráfico de barras com pontuação por critério
    fig_score, ax_score = plt.subplots(figsize=(7, 4))
    if len(quadro) > 0:
        sns.barplot(data=quadro, x='Critério', y='Pontos', palette='crest', ax=ax_score)
        ax_score.set_ylim(0, 20)
        ax_score.set_title('Pontuação por Critério (0–20)')
        ax_score.set_ylabel('Pontos')
        ax_score.set_xlabel('')
        ax_score.tick_params(axis='x', rotation=45)
    st.pyplot(fig_score, clear_figure=True)


    st.subheader('Mapa de Correlação')
    corr = df.corr()
    fig2, ax = plt.subplots()
    rotulos=["Preço A+1","Previsão","Eólica NE","Hidráulica SE","Carga SIN","Preço M+1","ena total","ENA Amazonas","ENA Grande","ENA Iguaçu","ENA PAraná","PLD SE","Câmbio USD/BRL","Térmica SIN"]
    sns.heatmap(corr, annot=True, cmap='crest',fmt='.1f', xticklabels=rotulos, yticklabels=rotulos, vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig2)

    st.subheader('Séries Auxiliares e Volatilidade')
    aux_cols = [col for col in df.columns if col != 'Price']
    selected = st.selectbox('Escolha a série', aux_cols)
    fig3 = px.line(x=df.index, y=df[selected], labels={'x': 'Data', 'y': selected}, title=f'Série de {selected}')
    st.plotly_chart(fig3, use_container_width=True)
    vol = df[selected].rolling(window=10).std()
    fig4 = px.line(x=vol.index, y=vol.values, labels={'x': 'Data', 'y': 'Volatilidade'}, title=f'Volatilidade de {selected} (std móvel 10)')
    st.plotly_chart(fig4, use_container_width=True)

else:
    st.error('Dados inválidos ou a coluna Price não está presente no arquivo.')


st.markdown('---')
st.caption('Este dashboard é um exemplo robusto que trata de erros de formatação ao carregar os dados e integra o modelo ARIMAX.')

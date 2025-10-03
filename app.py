import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Sectors and companies (defined early to avoid NameError)
sectors = {
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'BANKBARODA.NS', 'PNB.NS', 'CANBK.NS', 'UNIONBANK.NS', 'INDUSINDBK.NS', 'IDFCFIRSTB.NS', 'FEDERALBNK.NS', 'MAHABANK.NS', 'IOB.NS', 'EQUITASBNK.NS'],
    'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'LTIM.NS', 'TECHM.NS', 'MINDTREE.NS', 'LTTS.NS', 'TATAELXSI.NS', 'KPITTECH.NS', 'HAPPSTMNDS.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'OFSS.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'DRREDDY.NS', 'TORNTPHARM.NS', 'ZYDUSLIFE.NS', 'LUPIN.NS', 'ALKEM.NS', 'ABBOTINDIA.NS', 'AUROPHARMA.NS', 'GLENMARK.NS', 'IPCALAB.NS', 'LAURUSLABS.NS', 'BIOCON.NS', 'MANKIND.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'HEROMOTOCO.NS', 'BOSCHLTD.NS', 'MOTHERSON.NS', 'ASHOKLEY.NS', 'EXIDEIND.NS', 'SONACOMS.NS', 'UNOMINDA.NS', 'BHARATFORG.NS', 'TIINDIA.NS'],
    'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'GODREJCP.NS', 'VBL.NS', 'COLPAL.NS', 'MARICO.NS', 'DABUR.NS', 'TATACONSUM.NS', 'EMAMI.NS', 'JYOTHYLAB.NS', 'RADICO.NS', 'UBL.NS', 'MCDOWELL-N.NS'],
    'Energy': ['RELIANCE.NS', 'NTPC.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'COALINDIA.NS', 'POWERGRID.NS', 'TATAPOWER.NS', 'GAIL.NS', 'HINDPETRO.NS', 'ADANIPOWER.NS', 'NHPC.NS', 'JSWENERGY.NS', 'PETRONET.NS', 'BHEL.NS'],
    'Metals': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'NALCO.NS', 'SAIL.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'HINDZINC.NS', 'APLAPOLLO.NS', 'JSL.NS', 'MOIL.NS', 'GRSE.NS', 'HINDCOPPER.NS', 'WELCORP.NS'],
    'Telecom': ['BHARTIARTL.NS', 'TATACOMM.NS', 'TEJASNET.NS', 'HFCL.NS', 'TTML.NS', 'MTNL.NS', 'RAILTEL.NS', 'ITI.NS', 'VINDHYATEL.NS', 'HATHWAY.NS', 'GTLINFRA.NS', 'ONMOBILE.NS', 'NELCO.NS', 'EXICOM.NS', 'AVANTEL.NS'],
    'Real Estate': ['DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS', 'SOBHA.NS', 'BRIGADE.NS', 'PHOENIXLTD.NS', 'MINDSPACE.NS', 'ANANTRAJ.NS', 'MAHLIFE.NS', 'KALPATARU.NS', 'SUNCLAY.NS', 'INDIAGLYCOLS.NS', 'JKCEMENT.NS', 'JKLAKSHMI.NS'],
    'Insurance': ['LIC.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'ICICIPRULI.NS', 'ICICIGI.NS', 'GICRE.NS', 'NIACL.NS', 'STARHEALTH.NS', 'NIVA.NS', 'SBIINS.NS', 'RELIANCE.NS', 'BAJAJFINSV.NS', 'CHOLAFIN.NS', 'MAHINDRAFIN.NS', 'POWERFIN.NS'],
    'Consumer Durables': ['TITAN.NS', 'ASIANPAINT.NS', 'DIXON.NS', 'HAVELLS.NS', 'WHIRLPOOL.NS', 'SYMPHONY.NS', 'VGUARD.NS', 'POLYCAB.NS', 'BLUESTARCO.NS', 'BATAINDIA.NS', 'CENTURYPLY.NS', 'FINCABLES.NS', 'KEI.NS', 'RRKABEL.NS', 'JYOTILIFE.NS'],
    'Healthcare': ['APOLLOHOSP.NS', 'MAXHEALTH.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'SUNPHARMA.NS', 'CIPLA.NS', 'TORNTPHARM.NS', 'ZYDUSLIFE.NS', 'ALKEM.NS', 'ABBOTINDIA.NS', 'AUROPHARMA.NS', 'GLENMARK.NS', 'NH.NS', 'FORTIS.NS', 'KIMS.NS'],
    'Chemicals': ['PIDILITIND.NS', 'SRF.NS', 'AARTIIND.NS', 'DEEPAKNTRL.NS', 'VINATIORGA.NS', 'TATACHEM.NS', 'UPL.NS', 'ASTRAL.NS', 'ATUL.NS', 'ALEMBIC.NS', 'BALAMINES.NS', 'GUJALKALI.NS', 'ALKALI.NS', 'NAVKAR.NS', 'KANSAINER.NS'],
    'Media': ['ZEEMEDIA.NS', 'TVTODAY.NS', 'SUNTV.NS', 'PVRINOX.NS', 'INOXLEISURE.NS', 'BALAJITELE.NS', 'HTMEDIA.NS', 'SITI.NS', 'BAGFILMS.NS', 'EIHASSO.NS', 'SABTV.NS', 'CINEVISTA.NS', 'KSS.NS', 'DOLAT.NS', 'NAVKARSP.NS'],
    'Engineering': ['LT.NS', 'BHEL.NS', 'ABB.NS', 'SIEMENS.NS', 'CGPOWER.NS', 'THERMAX.NS', 'GRSE.NS', 'MAZDOCK.NS', 'BEL.NS', 'HAL.NS', 'BEML.NS', 'HBLENGINEERS.NS', 'KPITTECH.NS', 'TTML.NS', 'PTCINDIA.NS'],
    'Textiles': ['PAGEIND.NS', 'KPRMILL.NS', 'VARDHACRLC.NS', 'LMW.NS', 'ARVIND.NS', 'RAYMOND.NS', 'TRITURBINE.NS', 'GOKEX.NS', 'SUTLEJ.NS', 'NDL.NS', 'BSE.NS', 'PARAGMILK.NS', 'VTL.NS', 'RSWM.NS', 'SPLPETRO.NS'],
    'Infrastructure': ['LT.NS', 'ADANIPORTS.NS', 'IRB.NS', 'NCC.NS', 'GMRINFRA.NS', 'RVNL.NS', 'KEC.NS', 'PNCINFRA.NS', 'HCC.NS', 'JINDRILL.NS', 'GPPL.NS', 'DBL.NS', 'ITD.NS', 'JKIL.NS', 'ENGINERSIN.NS'],
    'Defence': ['HAL.NS', 'BEL.NS', 'MAZDOCK.NS', 'GRSE.NS', 'BEML.NS', 'DATAPATTNS.NS', 'BDL.NS', 'ASTRAMICRO.NS', 'MTARTECH.NS', 'PRECAM.NS', 'SPSDE.NS', 'TANEJAERO.NS', 'AVANTEL.NS', 'UNIMECH.NS', 'PTCINDIA.NS'],
    'Renewable Energy': ['ADANIGREEN.NS', 'TATAPOWER.NS', 'JSWENERGY.NS', 'NHPC.NS', 'SUZLON.NS', 'INOXWIND.NS', 'KPIGREEN.NS', 'ORIENTGREEN.NS', 'WEBSOL.NS', 'BORORENEW.NS', 'INSOLATION.NS', 'RAJRENEW.NS', 'URJA.NS', 'KAYNES.NS', 'WAAREE.NS']
}

def fetch_stock_data(symbol, period='1y'):
    """Fetch historical data for a stock."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            return None
        data.index = data.index.tz_localize(None)  # Remove timezone
        return data
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return None

def get_current_price(symbol):
    """Get real-time current price."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return info.get('currentPrice', info.get('regularMarketPrice', None))
    except:
        return None

def forecast_returns(data, periods=[7, 30, 365]):
    """Forecast % returns using Prophet for given periods (days)."""
    if data is None or len(data) < 100:  # Need sufficient data
        return {p: 0 for p in periods}
    
    # Prepare data for Prophet
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Fit model
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    
    # Forecast
    future = model.make_future_dataframe(periods=max(periods))
    forecast = model.predict(future)
    
    # Calculate % returns from last close
    last_close = df['y'].iloc[-1]
    projections = {}
    for p in periods:
        if len(forecast) > len(df) + p - 1:
            future_close = forecast['yhat'].iloc[len(df) + p - 1]
        else:
            future_close = last_close
        proj_return = ((future_close - last_close) / last_close) * 100
        projections[p] = round(proj_return, 2)
    
    return projections

def get_sector_avg_forecast(sector):
    """Average forecasts for sector companies."""
    projections = {7: [], 30: [], 365: []}
    for symbol in sectors[sector]:
        data = fetch_stock_data(symbol)
        if data is not None:
            fc = forecast_returns(data)
            for p in projections:
                projections[p].append(fc.get(p, 0))
    if not projections[7]:  # Empty
        return {p: 0 for p in [7, 30, 365]}
    return {p: round(np.mean(projs), 2) for p, projs in projections.items()}

def get_recommendations():
    """AI Recommendations: Top 3 sectors/companies by avg projected year return."""
    sector_projs = {}
    for sector in sectors:
        sector_projs[sector] = get_sector_avg_forecast(sector).get(365, 0)
    
    # Top sectors
    top_sectors = sorted(sector_projs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Top companies (flatten all, get top 3 by year proj)
    all_comp_projs = []
    for sector in sectors:
        for symbol in sectors[sector]:
            data = fetch_stock_data(symbol)
            if data is not None:
                proj = forecast_returns(data).get(365, 0)
                all_comp_projs.append((symbol, proj))
    top_companies = sorted(all_comp_projs, key=lambda x: x[1], reverse=True)[:3]
    
    return top_sectors, top_companies

def plot_candlestick(data, symbol):
    """Plot interactive candlestick chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{symbol} Candlestick (1Y)', 'Volume'),
                        row_width=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'],
                                 name='Candlestick'),
                  row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='blue'), row=2, col=1)
    
    fig.update_layout(title=f'{symbol} Analysis', xaxis_rangeslider_visible=False,
                      height=600, showlegend=False, template='plotly_white')
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig

def plot_projections(projections, title):
    """Line graph for projections."""
    periods = ['Week', 'Month', 'Year']
    values = [projections.get(7, 0), projections.get(30, 0), projections.get(365, 0)]
    fig = go.Figure(data=go.Scatter(x=periods, y=values, mode='lines+markers',
                                    line=dict(color='green', width=3),
                                    marker=dict(size=8)))
    fig.update_layout(title=title, yaxis_title='% Return', height=400)
    return fig

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main {background-color: #f0f8ff;}
    .stSelectbox > label {font-weight: bold; color: #1f77b4;}
    .metric {background-color: #e6f3ff; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title(" Indian Stock Market Analyzer")
st.markdown("Real-time sector-wise AI recommendations & projections for BSE/NSE (10+ Sectors)")

# Sidebar (now sectors is defined above)
st.sidebar.header("Select Options")
selected_sector = st.sidebar.selectbox("Choose Sector:", list(sectors.keys()))
selected_company = st.sidebar.selectbox("Choose Company:", sectors[selected_sector])

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "🔮 Projections", "📈 Charts"])

with tab1:
    st.subheader("AI Investment Recommendations (Upcoming Periods)")
    top_sectors, top_companies = get_recommendations()
    
    st.markdown("### Top 3 Sectors to Invest")
    sector_df = pd.DataFrame(top_sectors, columns=['Sector', 'Projected Year % Return'])
    st.table(sector_df.style.format({'Projected Year % Return': '{:.2f}%'}))
    
    st.markdown("### Top 3 Companies to Invest")
    comp_df = pd.DataFrame(top_companies, columns=['Company', 'Projected Year % Return'])
    st.table(comp_df.style.format({'Projected Year % Return': '{:.2f}%'}))

with tab2:
    st.subheader(f"Projections for {selected_company}")
    data = fetch_stock_data(selected_company)
    if data is not None:
        projections = forecast_returns(data)
        current_price = get_current_price(selected_company)
        st.metric("Current Price (₹)", current_price)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Week Return", f"{projections[7]}%")
        with col2:
            st.metric("Month Return", f"{projections[30]}%")
        with col3:
            st.metric("Year Return", f"{projections[365]}%")
        
        st.plotly_chart(plot_projections(projections, f"{selected_company} Projected Returns"), use_container_width=True)
        
        # Sector avg
        sector_proj = get_sector_avg_forecast(selected_sector)
        st.subheader(f"Sector Avg Projections: {selected_sector}")
        st.plotly_chart(plot_projections(sector_proj, f"{selected_sector} Sector Projections"), use_container_width=True)
    else:
        st.warning("No data available for this company. Try another.")

with tab3:
    st.subheader(f"Charts for {selected_company}")
    data = fetch_stock_data(selected_company)
    if data is not None:
        st.plotly_chart(plot_candlestick(data, selected_company), use_container_width=True)
    else:
        st.warning("No data for charts. Try another company.")

# Footer
st.markdown("---")
st.caption("Data: yfinance | Forecasts: Prophet | Not financial advice. Refresh for real-time.")

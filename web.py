import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import datetime
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder

# =====================================================================
# 1. UI 디자인 및 다크모드 스타일 설정
# =====================================================================
st.set_page_config(page_title="QUANT MONITOR", layout="wide")

st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
        background-color: #0B0E14 !important;
        color: #D1D4DC !important;
    }
    
    #MainMenu, header, footer {visibility: hidden;}
    
    .custom-navbar {
        position: fixed; top: 0; left: 0; width: 100%; height: 60px;
        background-color: rgba(11, 14, 20, 0.85); backdrop-filter: blur(12px);
        border-bottom: 1px solid #2A2E39; display: flex; align-items: center;
        padding: 0 2rem; z-index: 99999;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .custom-navbar b { letter-spacing: 1px; font-weight: 800; color: #E0E3EB; }
    .block-container { padding-top: 5.5rem !important; }
    
    /* 기존 3개 지표 카드 스타일 */
    div[data-testid="stMetric"] {
        background-color: #131722 !important;
        border-radius: 12px; padding: 1.2rem !important;
        border: 1px solid #2A2E39 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
        transition: all 0.2s ease-in-out;
        min-height: 120px; display: flex; flex-direction: column; justify-content: center;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.5) !important;
        border-color: #3C4353 !important;
    }
    div[data-testid="stMetricLabel"] { color: #787B86 !important; font-weight: 600 !important; font-size: 0.9rem !important; }
    div[data-testid="stMetricValue"] { color: #E0E3EB !important; font-weight: 800 !important; }
    
    .custom-metric {
        background-color: #131722; border-radius: 12px; padding: 1.2rem;
        border: 1px solid #2A2E39; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        min-height: 120px; display: flex; flex-direction: column; justify-content: space-between;
        transition: all 0.2s ease-in-out;
    }
    .custom-metric:hover {
        transform: translateY(-3px); border-color: #3C4353; box-shadow: 0 8px 15px rgba(0,0,0,0.5);
    }
    .metric-header { display: flex; justify-content: space-between; align-items: flex-start; }
    .metric-label { font-size: 0.9rem; color: #787B86; font-weight: 600; }
    .metric-value-container { margin-top: 0.8rem; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #E0E3EB; }
    .metric-delta { font-size: 1rem; font-weight: 600; margin-top: 0.2rem; }

    /* AI 분석 카드 전용 스타일 (심플화) */
    .ai-analysis-card {
        background: linear-gradient(145deg, #1A1E26 0%, #131722 100%);
        border: 1px solid #2A2E39;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease-in-out;
    }
    .ai-analysis-card:hover {
        border-color: #3C4353;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    .ai-title-group {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .ai-title {
        font-size: 1.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2962FF 0%, #1E46B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .ai-stars {
        color: #FFD700;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(255,215,0,0.3);
    }
    .ai-stat-group {
        display: flex;
        gap: 3rem;
    }
    .ai-stat {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.2rem;
    }
    .ai-stat-label {
        font-size: 0.8rem;
        color: #787B86;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .ai-stat-value {
        font-size: 1.1rem;
        color: #E0E3EB;
        font-weight: 800;
    }
    
    div[data-baseweb="tab-list"] { background-color: transparent; border-bottom: 1px solid #2A2E39; gap: 30px; }
    div[data-baseweb="tab"] { color: #787B86 !important; background-color: transparent !important; padding-bottom: 12px !important; }
    div[aria-selected="true"] { color: #E0E3EB !important; font-weight: 700 !important; border-bottom-color: #2962FF !important; }
    div[data-baseweb="select"] > div { background-color: #131722 !important; border-color: #2A2E39 !important; color: #E0E3EB !important; border-radius: 8px !important; }
    div[data-baseweb="select"]:hover > div { border-color: #3C4353 !important; }
    div[role="radiogroup"] label { color: #D1D4DC !important; }
    </style>
    <div class="custom-navbar"><b style="font-size:1.2rem;">QUANT MONITOR</b></div>
""", unsafe_allow_html=True)

# =====================================================================
# 2. 숫자 포맷 함수
# =====================================================================
def format_volume(value):
    if value >= 100000000: return f"{value / 100000000:.1f}억 주"
    elif value >= 10000: return f"{value / 10000:.1f}만 주"
    return f"{int(value):,} 주"

def format_amount(value):
    if value >= 1000000000000: return f"{value / 1000000000000:.2f}조 원"
    elif value >= 100000000: return f"{value / 100000000:.1f}억 원"
    elif value >= 10000: return f"{value / 10000:.1f}만 원"
    return f"{int(value):,} 원"

# =====================================================================
# 3. 데이터 로드 
# =====================================================================
@st.cache_data(ttl=600)
def load_market_data():
    try:
        codes = fdr.StockListing('KOSPI200')['Code'].tolist() + fdr.StockListing('KOSDAQ150')['Code'].tolist()
    except:
        codes = fdr.StockListing('KRX').sort_values(by='Marcap', ascending=False)['Code'].tolist()[:350]
    df = fdr.StockListing('KRX')
    df = df[df['Code'].isin(codes)].copy()
    df['Amount'] = df['Close'] * df['Volume']
    df['ChgRate'] = (df['Changes'] / (df['Close'] - df['Changes'] + 1e-9)) * 100 if 'Changes' in df.columns else 0.0
    return df

df = load_market_data()

# =====================================================================
# 4. 메인 화면 및 탭 구성
# =====================================================================
st.markdown("<br>", unsafe_allow_html=True)
col_search, _ = st.columns([1, 2])
with col_search:
    stock_name = st.selectbox("종목 검색", ["종목을 선택하세요"] + df['Name'].tolist(), label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
tab_chart, tab_rank = st.tabs(["주가 차트 분석", "시장 거래 동향"])

with tab_chart:
    if stock_name != "종목을 선택하세요":
        info = df[df['Name'] == stock_name].iloc[0]
        
        end = datetime.date.today()
        start = end - datetime.timedelta(days=365)
        hist = fdr.DataReader(info['Code'], start, end)

        if not hist.empty:
            # -----------------------------------------------------
            # 1) 기본 3개 지표 카드
            # -----------------------------------------------------
            c1, c2, c3 = st.columns(3)
            
            is_up = info['ChgRate'] >= 0
            chg_color = "#F23645" if is_up else "#2962FF" 
            chg_arrow = "▲" if is_up else "▼"
            
            custom_card_html = f"""
            <div class="custom-metric">
                <div class="metric-header">
                    <span class="metric-label">현재가</span>
                </div>
                <div class="metric-value-container">
                    <div class="metric-value">{int(info['Close']):,} 원</div>
                    <div class="metric-delta" style="color: {chg_color};">{chg_arrow} {abs(info['ChgRate']):.2f}%</div>
                </div>
            </div>
            """
            c1.markdown(custom_card_html, unsafe_allow_html=True)
            c2.metric("당일 거래량", format_volume(info['Volume']))
            c3.metric("당일 거래대금", format_amount(info['Amount']))

            # -----------------------------------------------------
            # 2) AI 분석 카드 섹션 (심플 버전)
            # -----------------------------------------------------
            ai_card_html = f"""
            <div class="ai-analysis-card">
                <div class="ai-title-group">
                    <div class="ai-title">AI 분석</div>
                    <div class="ai-stars">★★★★<span style="color:#3C4353;">★</span></div>
                </div>
                <div class="ai-stat-group">
                    <div class="ai-stat">
                        <span class="ai-stat-label">투자 의견</span>
                        <span class="ai-stat-value" style="color: #F23645;">매수 우세</span>
                    </div>
                    <div class="ai-stat">
                        <span class="ai-stat-label">상승 확률</span>
                        <span class="ai-stat-value">78%</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(ai_card_html, unsafe_allow_html=True)

            # -----------------------------------------------------
            # 3) 차트 렌더링 (구조 유지, 다크 디자인 적용)
            # -----------------------------------------------------
            fig = go.Figure()
            
            # 주가 라인 (그라데이션 유지)
            fill_color = "rgba(242, 54, 69, 0.08)" if is_up else "rgba(41, 98, 255, 0.08)"
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['Close'], mode='lines', name='종가', 
                line=dict(color=chg_color, width=2.5), fill='tozeroy', fillcolor=fill_color, connectgaps=True,
                hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:,}원</b><extra></extra>"
            ))
            
            chart_start = hist.index.min()
            chart_end = hist.index.max()
            pad = pd.Timedelta(days=2)

            fig.update_layout(
                dragmode="zoom", hovermode="x unified",
                template="plotly_dark", 
                plot_bgcolor="#131722", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10), height=450,
                xaxis=dict(
                    showgrid=True, gridcolor="#2A2E39", showline=False, # 다크 테마용 은은한 그리드
                    rangebreaks=[dict(bounds=["sat", "mon"])],
                    range=[chart_start, chart_end], minallowed=chart_start - pad, maxallowed=chart_end + pad,
                    tickformatstops=[dict(dtickrange=[None, "M1"], value="%m월 %d일"), dict(dtickrange=["M1", None], value="%Y년 %m월")],
                    tickfont=dict(color="#787B86")
                ),
                yaxis=dict(showgrid=True, gridcolor="#2A2E39", fixedrange=True, tickformat=",", tickfont=dict(color="#787B86")),
                hoverlabel=dict(bgcolor="#131722", bordercolor="#2A2E39", font=dict(color="#D1D4DC")) # 다크 툴팁
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True})

    else:
        st.info("상단 검색창에서 종목을 선택해 주세요.")

# =====================================================================
# 6. 시장 랭킹 탭
# =====================================================================
with tab_rank:
    def draw_aggrid(data, title):
        st.markdown(f"<h4 style='font-size:1.1rem; color:#E0E3EB; margin-bottom:15px; font-weight:700;'>{title}</h4>", unsafe_allow_html=True)
        
        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_default_column(sortable=True, filter=False, resizable=False, flex=1)
        
        for c in ["현재가", "거래량", "거래대금"]:
            gb.configure_column(c, valueFormatter="value.toLocaleString()", type=["numericColumn"])
        
        go = gb.build()
        
        custom_css = {
            ".ag-root-wrapper": {"border": "1px solid #2A2E39", "border-radius": "8px"},
            ".ag-header": {"background-color": "#0B0E14", "border-bottom": "1px solid #2A2E39", "color": "#787B86"},
            ".ag-row": {"background-color": "#131722", "border-bottom": "1px solid #2A2E39", "color": "#D1D4DC"},
            ".ag-row-hover": {"background-color": "#1E222D !important"}, 
            ".ag-row-odd": {"background-color": "#0F131A"}, 
            ".ag-cell": {"line-height": "40px"}
        }
        
        AgGrid(
            data, gridOptions=go, height=600, 
            theme="alpine", custom_css=custom_css, columns_auto_size_mode=2 
        )

    df_rank = df[['Name', 'Close', 'Volume', 'Amount']].rename(columns={'Name':'종목명', 'Close':'현재가', 'Volume':'거래량', 'Amount':'거래대금'})
    c1, c2 = st.columns(2)
    with c1: draw_aggrid(df_rank.sort_values('거래대금', ascending=False).head(20), "거래대금 상위 20개 종목")
    with c2: draw_aggrid(df_rank.sort_values('거래량', ascending=False).head(20), "거래량 상위 20개 종목")
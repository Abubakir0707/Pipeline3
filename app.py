import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import requests

st.set_page_config(
    page_title="Lead Scoring & Pipeline Manager",
    page_icon="📊",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .priority-high  { background:#EAF3DE; color:#27500A; padding:2px 10px;
                      border-radius:20px; font-size:12px; font-weight:600; }
    .priority-medium{ background:#FpipAEEDA; color:#633806; padding:2px 10px;
                      border-radius:20px; font-size:12px; font-weight:600; }
    .priority-low   { background:#FCEBEB; color:#791F1F; padding:2px 10px;
                      border-radius:20px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Trello API helpers ────────────────────────────────────────────────────────
BASE = "https://api.trello.com/1"

def trello_get(path, params):
    r = requests.get(f"{BASE}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def trello_post(path, params):
    r = requests.post(f"{BASE}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def get_boards(key, token):
    return trello_get("/members/me/boards", {"key":key,"token":token,"fields":"id,name"})

def get_lists(board_id, key, token):
    return trello_get(f"/boards/{board_id}/lists", {"key":key,"token":token,"filter":"open","fields":"id,name"})

def create_card(list_id, name, desc, key, token):
    return trello_post("/cards", {"key":key,"token":token, "idList":list_id,"name":name,"desc":desc})

# ── Sample lead data ──────────────────────────────────────────────────────────
SAMPLE_DATA = pd.DataFrame({
    "company":        ["Apex SaaS","FinCore Ltd","MedLogic","RetailHub","NovaTech",
                       "GreenPay","CarePoint","ShopWave","LexPro","BuildCo",
                       "CloudNine","DataPulse","HealthFirst","StyleMart","TradeFin"],
    "industry":       ["SaaS","Finance","Healthcare","Retail","SaaS",
                       "Finance","Healthcare","Retail","Legal","Construction",
                       "SaaS","SaaS","Healthcare","Retail","Finance"],
    "budget_k":       [120,95,80,45,200,60,35,55,75,40,150,90,65,50,110],
    "employees":      [500,200,350,80,1200,150,60,120,180,90,600,300,250,100,400],
    "engagement_score":[88,76,71,54,91,47,32,63,70,41,85,79,58,49,68],
    "deal_stage":      ["Proposal","Demo","Discovery","Contacted","Proposal",
                       "Nurture","Aware","Demo","Discovery","Contacted",
                       "Proposal","Demo","Discovery","Nurture","Proposal"],
    "converted":      [1,1,1,0,1,0,0,1,1,0,1,1,0,0,1],
})

# ── ML model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    le_ind   = LabelEncoder().fit(df["industry"])
    le_stage = LabelEncoder().fit(df["deal_stage"])
    X = pd.DataFrame({
        "budget_k":         df["budget_k"],
        "employees":        df["employees"],
        "engagement_score": df["engagement_score"],
        "industry_enc":     le_ind.transform(df["industry"]),
        "deal_stage_enc":   le_stage.transform(df["deal_stage"]),
    })
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, df["converted"])
    return model, le_ind, le_stage

def score_leads(df, model, le_ind, le_stage):
    df = df.copy()
    # Handle Missing Values
    df['budget_k'] = pd.to_numeric(df['budget_k'], errors='coerce').fillna(0)
    df['employees'] = pd.to_numeric(df['employees'], errors='coerce').fillna(1)
    df['engagement_score'] = pd.to_numeric(df['engagement_score'], errors='coerce').fillna(0)
    
    df["industry_s"]   = df["industry"].apply(lambda x: x if x in le_ind.classes_ else le_ind.classes_[0])
    df["deal_stage_s"] = df["deal_stage"].apply(lambda x: x if x in le_stage.classes_ else le_stage.classes_[0])
    
    X = pd.DataFrame({
        "budget_k":         df["budget_k"],
        "employees":        df["employees"],
        "engagement_score": df["engagement_score"],
        "industry_enc":     le_ind.transform(df["industry_s"]),
        "deal_stage_enc":   le_stage.transform(df["deal_stage_s"]),
    })
    
    probs = model.predict_proba(X)[:, 1]
    df["conversion_prob"] = (probs * 100).round(1)
    df["priority"] = pd.cut(df["conversion_prob"], bins=[-1,40,65,101], labels=["Low","Medium","High"])
    df["strategy"] = df.apply(_strategy, axis=1)
    return df

def _strategy(row):
    p, stage = row["conversion_prob"], row.get("deal_stage","")
    if p >= 75: return "Executive outreach + custom proposal"
    if p >= 55: return "Follow-up with ROI calculator" if stage=="Demo" else "Product demo + case study"
    if p >= 35: return "Nurture sequence + trial offer"
    return "Drip email campaign"

# ── App Logic ─────────────────────────────────────────────────────────────────
if "df_scored" not in st.session_state:
    m, li, ls = train_model(SAMPLE_DATA)
    st.session_state.update({"df_scored": score_leads(SAMPLE_DATA,m,li,ls), "model":m, "le_ind":li, "le_stage":ls})

# Sidebar & Filtering
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Trello Connection
    auto_key = st.secrets.get("TRELLO_API_KEY", "")
    auto_token = st.secrets.get("TRELLO_TOKEN", "")
    t_key = st.text_input("Trello Key", value=auto_key, type="password")
    t_token = st.text_input("Trello Token", value=auto_token, type="password")
    
    if st.button("Connect Trello"):
        try:
            st.session_state.boards = get_boards(t_key, t_token)
            st.session_state.trello_connected = True
            st.session_state.t_key, st.session_state.t_token = t_key, t_token
            st.success("Connected!")
        except Exception as e: st.error(f"Error: {e}")

    # File Upload
    uploaded = st.file_uploader("Upload Lead Data", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
        st.session_state.df_scored = score_leads(raw, st.session_state.model, st.session_state.le_ind, st.session_state.le_stage)

# Main Dashboard
df_all = st.session_state.df_scored
st.title("📊 Lead Scoring Manager")

# Metrics
k1, k2, k3 = st.columns(3)
k1.metric("🔥 High Priority", len(df_all[df_all["priority"]=="High"]))
k2.metric("🎯 Avg Probability", f"{df_all['conversion_prob'].mean():.1f}%")
k3.metric("💰 Pipeline Value", f"${df_all['budget_k'].sum():,.0f}k")

# Pipeline Table
st.subheader("Lead Pipeline")
cols = ["company","industry","budget_k","conversion_prob","priority","strategy"]
st.dataframe(df_all[cols].style.background_gradient(subset=["conversion_prob"], cmap="RdYlGn"), use_container_width=True)

# Trello Integration
if st.session_state.get("trello_connected"):
    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        target_lead = st.selectbox("Select Lead to Push", df_all["company"])
        lead_data = df_all[df_all["company"] == target_lead].iloc[0]
    with c2:
        if st.button("🚀 Push to Trello"):
            desc = f"Strategy: {lead_data['strategy']}\nBudget: ${lead_data['budget_k']}k"
            # Logic to pick list ID from session state would go here
            st.toast(f"Pushed {target_lead} to Trello!")

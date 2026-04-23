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
    .trello-card    { background:#F0F4FF; border-left:4px solid #0052CC;
                      border-radius:6px; padding:10px 14px; margin-bottom:8px; }
    .trello-card h4 { margin:0 0 4px 0; font-size:14px; color:#172B4D; }
    .trello-card p  { margin:0; font-size:12px; color:#5E6C84; }
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
    return trello_get("/members/me/boards",
                      {"key":key,"token":token,"fields":"id,name"})

def get_lists(board_id, key, token):
    return trello_get(f"/boards/{board_id}/lists",
                      {"key":key,"token":token,"filter":"open","fields":"id,name"})

def get_cards(list_id, key, token):
    return trello_get(f"/lists/{list_id}/cards",
                      {"key":key,"token":token,
                       "fields":"id,name,desc,dateLastActivity,url"})

def create_card(list_id, name, desc, key, token):
    return trello_post("/cards",
                       {"key":key,"token":token,
                        "idList":list_id,"name":name,"desc":desc})

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
    "deal_stage":     ["Proposal","Demo","Discovery","Contacted","Proposal",
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
    
    # --- SAFETY FIX: Handle Missing Values ---
    df['budget_k'] = pd.to_numeric(df['budget_k'], errors='coerce').fillna(0)
    df['employees'] = pd.to_numeric(df['employees'], errors='coerce').fillna(1)
    df['engagement_score'] = pd.to_numeric(df['engagement_score'], errors='coerce').fillna(0)
    df['industry'] = df['industry'].fillna('SaaS')
    df['deal_stage'] = df['deal_stage'].fillna('Discovery')

    df["industry_s"]   = df["industry"].apply(
        lambda x: x if x in le_ind.classes_   else le_ind.classes_[0])
    df["deal_stage_s"] = df["deal_stage"].apply(
        lambda x: x if x in le_stage.classes_ else le_stage.classes_[0])
    
    X = pd.DataFrame({
        "budget_k":         df["budget_k"],
        "employees":        df["employees"],
        "engagement_score": df["engagement_score"],
        "industry_enc":     le_ind.transform(df["industry_s"]),
        "deal_stage_enc":   le_stage.transform(df["deal_stage_s"]),
    })
    
    probs = model.predict_proba(X)[:, 1]
    df["conversion_prob"] = (probs * 100).round(1)
    df["priority"] = pd.cut(df["conversion_prob"],
                            bins=[-1,40,65,101], labels=["Low","Medium","High"])
    df["strategy"] = df.apply(_strategy, axis=1)
    return df.drop(columns=["industry_s","deal_stage_s"], errors="ignore")

def _strategy(row):
    p, stage = row["conversion_prob"], row.get("deal_stage","")
    if p >= 75: return "Executive outreach + custom proposal"
    if p >= 55: return "Follow-up with ROI calculator" if stage=="Demo" else "Product demo + case study"
    if p >= 35: return "Nurture sequence + trial offer"
    return "Drip email campaign"

# ── Session state init ────────────────────────────────────────────────────────
if "df_scored" not in st.session_state:
    m, li, ls = train_model(SAMPLE_DATA)
    st.session_state.update({"df_scored": score_leads(SAMPLE_DATA,m,li,ls),
                              "model":m, "le_ind":li, "le_stage":ls})
if "trello_connected"    not in st.session_state: st.session_state.trello_connected    = False
if "selected_list_id"    not in st.session_state: st.session_state.selected_list_id    = None
if "selected_list_name"  not in st.session_state: st.session_state.selected_list_name  = "—"
if "chat_history"        not in st.session_state: st.session_state.chat_history        = []

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings & Filters")

    # Pull keys from Streamlit Secrets automatically
    auto_key = st.secrets.get("TRELLO_API_KEY", "")
    auto_token = st.secrets.get("TRELLO_TOKEN", "")

    with st.expander("🔑 Trello connection", expanded=not st.session_state.trello_connected):
        trello_key   = st.text_input("API Key",   value=auto_key, type="password")
        trello_token = st.text_input("API Token", value=auto_token, type="password")

        if st.button("Connect to Trello"):
            try:
                boards = get_boards(trello_key, trello_token)
                st.session_state.trello_connected = True
                st.session_state.trello_key   = trello_key
                st.session_state.trello_token = trello_token
                st.session_state.boards       = boards
                st.success(f"✅ Connected!")
            except Exception as e:
                st.error(f"❌ Failed: {e}")

    # Board / list picker
    if st.session_state.trello_connected:
        boards = st.session_state.boards
        chosen_board = st.selectbox("Board", [b["name"] for b in boards])
        board_id = next(b["id"] for b in boards if b["name"]==chosen_board)

        try:
            lists = get_lists(board_id, st.session_state.trello_key, st.session_state.trello_token)
            chosen_list = st.selectbox("List", [l["name"] for l in lists])
            st.session_state.selected_list_id   = next(l["id"] for l in lists if l["name"]==chosen_list)
            st.session_state.selected_list_name = chosen_list
        except:
            pass

    st.divider()

    uploaded = st.file_uploader("Upload lead CSV", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
        required = {"company","industry","budget_k","employees","engagement_score","deal_stage"}
        missing = required - set(raw.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.session_state.df_scored = score_leads(raw, st.session_state.model, st.session_state.le_ind, st.session_state.le_stage)

    df_all = st.session_state.df_scored
    industries   = st.multiselect("Industry", sorted(df_all["industry"].unique()), default=sorted(df_all["industry"].unique()))
    budget_range = st.slider("Budget ($k)", int(df_all["budget_k"].min()), int(df_all["budget_k"].max()), (0, 1000))
    priorities   = st.multiselect("Priority", ["High","Medium","Low"], default=["High","Medium","Low"])

# ── Main UI Logic ─────────────────────────────────────────────────────────────
df = df_all[
    df_all["industry"].isin(industries) &
    df_all["budget_k"].between(*budget_range) &
    df_all["priority"].isin(priorities)
].reset_index(drop=True)

st.title("📊 Lead Scoring Manager")
k1,k2,k3 = st.columns(3)
k1.metric("🔥 High-priority", int((df["priority"]=="High").sum()))
k2.metric("🎯 Avg Prob", f"{df['conversion_prob'].mean():.1f}%" if len(df) else "0%")
k3.metric("💰 Total Pipeline", f"${df['budget_k'].sum():,}k")

col_left, col_right = st.columns([2, 1], gap="medium")

with col_left:
    st.subheader("Lead pipeline")
    def color_prob(val):
        if val >= 70: return "color:#3B6D11;font-weight:600"
        return "color:#A32D2D;font-weight:600"

    cols = ["company","industry","budget_k","conversion_prob","priority","strategy"]
    styled = (df[cols].style.map(color_prob, subset=["conversion_prob"])
              .format({"conversion_prob":"{:.1f}%","budget_k":"${:,}k"}))
    st.dataframe(styled, width="stretch", height=400) # FIXED: width="stretch" for 2026

with col_right:
    st.subheader("📤 Push to Trello")
    if st.session_state.trello_connected and len(df) > 0:
        lead_pick = st.selectbox("Choose lead", df["company"])
        row = df[df["company"]==lead_pick].iloc[0]
        if st.button("Send to Trello"):
            try:
                create_card(st.session_state.selected_list_id, f"{row['company']} - {row['priority']}", row['strategy'], st.session_state.trello_key, st.session_state.trello_token)
                st.success("Card Created!")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Check Trello connection or lead filters.")

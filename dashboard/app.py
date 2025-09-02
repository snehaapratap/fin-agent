import os, requests, pandas as pd, streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic Financial Advisor", layout="wide")

st.title("🔮 Agentic Financial Advisor (CrewAI)")
syms = st.text_input("Symbols (comma-separated)", value=os.getenv("SYMBOLS","AAPL,MSFT,TSLA").strip())
capital = st.number_input("Capital (paper)", value=100000)

if st.button("Run Analysis & Propose Trades"):
    symbols = [s.strip().upper() for s in syms.split(',') if s.strip()]
    with st.spinner("Calling backend..."):
        r = requests.post(f"{BACKEND}/trade", json={"symbols": symbols, "capital": capital})
        if r.ok:
            data = r.json()
            st.subheader("Proposed Decisions")
            st.dataframe(pd.DataFrame(data.get("decisions", [])))
            st.subheader("Sentiment")
            st.json(data.get("sentiment", {}))
        else:
            st.error(r.text)
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from utils.schemas import AnalyzeRequest, AnalyzeResponse, TradeRequest, TradeResponse
from crew.coordinator import AdvisorCrew

load_dotenv()
app = FastAPI(title="Agentic Financial Advisor (CrewAI)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    crew = AdvisorCrew(req.symbols)
    out = crew.run()
    return AnalyzeResponse(insights=out["insights"], sentiment=out["sentiment"], news=out["news"])

@app.post("/trade", response_model=TradeResponse)
def trade(req: TradeRequest):
    crew = AdvisorCrew(req.symbols, capital=req.capital)
    out = crew.run()
    decisions = out.get("decisions", [])
    return TradeResponse(decisions=decisions, notes="Paper-only; risk rules applied.")

@app.get("/")
def root():
    return {"ok": True}

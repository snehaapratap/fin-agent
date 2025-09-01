import os
from crewai import Agent

LLM = os.getenv("MODEL_NAME", "gpt-4o-mini")

def news_analyst():
    return Agent(
        role="News Analyst",
        goal=("Identify and summarize the most market-moving news for the given symbols, "
              "focusing on catalysts, earnings, guidance, macro, and risks."),
        backstory=("You are a meticulous sell-side analyst known for concise summaries and signal extraction."),
        verbose=True,
        llm=LLM,
        allow_delegation=False,
    )

def sentiment_agent():
    return Agent(
        role="Sentiment Analyst",
        goal=("Aggregate sentiment across curated news and social snippets and output a per-symbol score in [-1,1]."),
        backstory=("You specialize in NLP sentiment and event polarity for financial text."),
        verbose=True,
        llm=LLM,
        allow_delegation=False,
    )

def trader_agent():
    return Agent(
        role="Trader",
        goal=("Propose trade actions (buy/sell/hold) with sizes and concise reasoning. Respect risk constraints."),
        backstory=("You are a systematic trader blending RL signals with fundamentals and sentiment."),
        verbose=True,
        llm=LLM,
        allow_delegation=False,
    )

def risk_manager_agent():
    return Agent(
        role="Risk Manager",
        goal=("Enforce risk constraints, cap position sizes, and block trades violating rules."),
        backstory=("You manage exposure, stops, and drawdown limits."),
        verbose=True,
        llm=LLM,
        allow_delegation=False,
    )

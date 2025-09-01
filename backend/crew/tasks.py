from crewai import Task

news_task_template = (
    "Given symbols: {symbols}. Using the provided news items, produce a compact brief per symbol: "
    "top catalysts, risks, and a 1-sentence implication. Keep to bullet points.")

sentiment_task_template = (
    "Given symbols: {symbols} and the aggregated headlines/snippets, compute a sentiment score per symbol in [-1,1] "
    "and a one-liner rationale. Output JSON: {symbol: {score: float, reason: str}}.")

trader_task_template = (
    "Given prices, indicators, and sentiment JSON, propose actions (buy/sell/hold) and sizes. "
    "Respect risk constraints. Output JSON list: [{symbol, action, size, reason}].")

risk_task_template = (
    "Validate proposed trades against risk rules (max position %, exposure, sanity checks). "
    "Return the approved list with any adjustments and a brief note.")


def news_task(agent, symbols, context):
    return Task(description=news_task_template.format(symbols=symbols), agent=agent, context=context)

def sentiment_task(agent, symbols, context):
    return Task(description=sentiment_task_template.format(symbols=symbols), agent=agent, context=context)

def trader_task(agent, context):
    return Task(description=trader_task_template, agent=agent, context=context)

def risk_task(agent, context):
    return Task(description=risk_task_template, agent=agent, context=context)

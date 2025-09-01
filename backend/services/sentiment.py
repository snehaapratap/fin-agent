from transformers import pipeline

# Load once (CPU ok for demo)
sent_pipe = pipeline("sentiment-analysis")

def score_texts(texts):
    out = sent_pipe(texts[:32])  # batch cap for demo
    # Return average polarity in [-1,1]
    score = 0.0
    for o in out:
        s = o['score'] if o['label'].upper().startswith('POS') else -o['score']
        score += s
    return score / max(1, len(out))

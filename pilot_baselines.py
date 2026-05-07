"""
Three Off-the-Shelf Sentiment Baselines on the 18-item Singlish Probe Set.

Baselines:
  1. VADER (lexicon-based, English)             
  2. DistilBERT-SST-2 (binary English)          
  3. XLM-R-twitter-sentiment (multilingual 3-way)
"""
import warnings
warnings.filterwarnings("ignore")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# 18-item Singlish Probe Set
TEST_SET = [
    # H1: Particle-Induced Sentiment Inversion
    {"id": 1,  "text": "The new MRT line damn good leh, so fast reach already.",
     "gold": "pos", "probe": "H1_particle"},
    {"id": 2,  "text": "The new MRT line damn good meh? I wait 20 mins just now.",
     "gold": "neg", "probe": "H1_particle"},
    {"id": 3,  "text": "Food court food can lah, not bad.",
     "gold": "pos", "probe": "H1_particle"},
    {"id": 4,  "text": "Food court food can meh? Taste like cardboard sia.",
     "gold": "neg", "probe": "H1_particle"},
    {"id": 5,  "text": "Walao the queue so long. Shiok sia after 1 hour finally got my chendol.",
     "gold": "pos", "probe": "H1_particle"},
    # H2: Code-Switched Sentiment Markers
    {"id": 6,  "text": "Bo liao la, every day also complain about the weather.",
     "gold": "neg", "probe": "H2_codeswitch"},
    {"id": 7,  "text": "Kopi at this place really power sia, best in neighbourhood.",
     "gold": "pos", "probe": "H2_codeswitch"},
    {"id": 8,  "text": "The service damn jialat, waited 45 mins for one plate chicken rice.",
     "gold": "neg", "probe": "H2_codeswitch"},
    {"id": 9,  "text": "Wah, the prata there steady pom pi pi, must try.",
     "gold": "pos", "probe": "H2_codeswitch"},
    {"id": 10, "text": "Aiyoh the ERP increase again, sibei sian.",
     "gold": "neg", "probe": "H2_codeswitch"},
    # H3: Sarcasm via Discourse Particles
    {"id": 11, "text": "Wah government so generous, give us $300 voucher lor.",
     "gold": "neg", "probe": "H3_sarcasm"},
    {"id": 12, "text": "Ya lah ya lah, PAP solve everything one.",
     "gold": "neg", "probe": "H3_sarcasm"},
    {"id": 13, "text": "HDB prices drop? Sure anot? First time hearing this one.",
     "gold": "neg", "probe": "H3_sarcasm"},
    {"id": 14, "text": "Best lah our public transport, cancel my bus for the third time today.",
     "gold": "neg", "probe": "H3_sarcasm"},
    {"id": 15, "text": "Confirm plus chop this hawker stall is the real deal, queued 2 times also worth it.",
     "gold": "pos", "probe": "H3_sarcasm"},
    # Standard-English Controls
    {"id": 16, "text": "I love this place so much, will come back again.",
     "gold": "pos", "probe": "control"},
    {"id": 17, "text": "Terrible experience, would not recommend.",
     "gold": "neg", "probe": "control"},
    {"id": 18, "text": "It's okay, nothing special.",
     "gold": "neu", "probe": "control"},
]

# Load Models
print("Loading models (first run downloads ~1GB)...")
vader = SentimentIntensityAnalyzer()
distilbert = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
xlmr = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
)

# Label-Mapping Helpers
def vader_label(text):
    c = vader.polarity_scores(text)["compound"]
    if c >= 0.05:  return "pos"
    if c <= -0.05: return "neg"
    return "neu"

def db_label(text):
    # DistilBERT-SST-2 is Binary: Maps POSITIVE -> pos, NEGATIVE -> neg
    return "pos" if distilbert(text)[0]["label"] == "POSITIVE" else "neg"

XR_MAP = {
    "Positive": "pos", "positive": "pos",
    "Negative": "neg", "negative": "neg",
    "Neutral":  "neu", "neutral":  "neu",
}
def xr_label(text):
    raw = xlmr(text)[0]["label"]
    return XR_MAP.get(raw, raw.lower()[:3])

# Run Baselines on Every Probe Item
rows = []
for item in TEST_SET:
    rows.append({
        **item,
        "vader": vader_label(item["text"]),
        "distilbert": db_label(item["text"]),
        "xlmr": xr_label(item["text"]),
    })

# Per-Item Results
print(f"\n{'id':>2}  {'probe':<14} {'gold':<4} {'vader':<6} {'distilB':<8} {'xlmr':<6}  text")
print("-" * 120)
for r in rows:
    mark = lambda m: "OK" if r[m] == r["gold"] else "x "
    # DistilBERT can't predict "neu" so treat neu items as not-applicable
    db_cell = "n/a    " if r["gold"] == "neu" else f"{r['distilbert']}{mark('distilbert')}"
    print(f"{r['id']:>2}  {r['probe']:<14} {r['gold']:<4} "
          f"{r['vader']}{mark('vader'):<2}    "
          f"{db_cell:<8} "
          f"{r['xlmr']}{mark('xlmr'):<2}  "
          f"{r['text']}")
        
# Accuracy by Probe Stratum (DistilBERT excluded on neutral items)
def accuracy(rs, model_key):
    if model_key == "distilbert":
        rs = [r for r in rs if r["gold"] in ("pos", "neg")]
    correct = sum(1 for r in rs if r[model_key] == r["gold"])
    return correct, len(rs)

print("\n=== Accuracy by probe stratum ===")
for probe in ("H1_particle", "H2_codeswitch", "H3_sarcasm", "control"):
    sub = [r for r in rows if r["probe"] == probe]
    parts = []
    for m in ("vader", "distilbert", "xlmr"):
        c, n = accuracy(sub, m)
        parts.append(f"{m} {c}/{n}")
    print(f"  {probe:<15} " + "  ".join(parts))

print("\n=== Overall accuracy ===")
for m in ("vader", "distilbert", "xlmr"):
    c, n = accuracy(rows, m)
    print(f"  {m:<11} {c}/{n} = {c/n:.1%}")
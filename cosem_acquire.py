"""
CoSEM Sampling script for COMP5840M: Data Mining & Text Analytics Project.

  1. Clones the public CoSEM v5 Corpus from GitHub (CC-licensed, anonymised)
  2. Parses the Corpus's tagged-utterance format
  3. Filters out utterances unsuitable for sentiment annotation (i.e. too short, links-only, sentiment-neutral, etc.)
  5. Outputs two CSVs:
       cosem_pilot_sample.csv     -- 100 utterances for pilot/fine-tune
       cosem_main_pool.csv         -- 2000 utterances pool for main-study annotation
"""
import os
import re
import random
import csv
import zipfile
import subprocess
import shutil

REPO_URL = "https://github.com/wdwgonzales/CoSEM.git"
CLONE_DIR = "CoSEM_repo"
ZIP_NAME = "corpus_COSEM_v5.zip"
EXTRACTED_DIR = "CoSEM_v5"

PILOT_N = 100
MAIN_POOL_N = 2000     # over-sample: annotate 1500 and keep 500 as buffer
RANDOM_SEED = 42       # fixed for reproducibility

# Stratified-Sampling Target Proportions for the Pilot
HIGH_SIGNAL_PROPORTION = 0.70

# Step 1: Clone the CoSEM Corpus
if not os.path.isdir(CLONE_DIR):
    print("Cloning CoSEM repository (~22MB)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR],
        check=True
    )
else:
    print(f"Reusing existing {CLONE_DIR}/")

# Step 2: Unzip the CoSEM Corpus
zip_path = os.path.join(CLONE_DIR, ZIP_NAME)
extract_to = os.path.join(CLONE_DIR, EXTRACTED_DIR)
if not os.path.isdir(extract_to):
    print(f"Extracting {ZIP_NAME}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(CLONE_DIR)
else:
    print(f"Reusing existing {extract_to}/")

# Step 3: Parse all Chunks
TAG_RE = re.compile(r"<<COSEM:([^>]+)>>\s*(.*)")
URL_RE = re.compile(r"https?://\S+")

def parse_tag(tag):
    """
    Tag format: 17CF01-1-21SGCHF00-2016
      17CF01   = year(17)+ethnicity(C)+gender(F)+chat(01)
      1        = utterance line number
      21SGCHF00= age(21)+nationality(SG)+ethnicity(CH)+gender(F)+suffix
      2016     = year of utterance
    We extract year, age, ethnicity, gender of the speaker.
    """
    parts = tag.split("-")
    speaker = parts[2] if len(parts) > 2 else ""
    year = parts[3] if len(parts) > 3 else ""
    m = re.match(r"(\d+)([A-Z]{2})([A-Z]{2})([A-Z])", speaker)
    if m:
        age, nat, eth, gen = m.groups()
        return {"speaker_age": age, "nationality": nat,
                "ethnicity": eth, "gender": gen, "year": year}
    return {"speaker_age": "", "nationality": "", "ethnicity": "",
            "gender": "", "year": year}

def is_usable(text):
    """Filter heuristics for sentiment-annotation suitability."""
    if not text or len(text.strip()) == 0:
        return False
    text = text.strip()
    # Length: too short = likely backchannel ('ok', 'lol'); too long = likely multiple thoughts, harder to single-label
    if len(text) < 40 or len(text) > 280:
        return False
    # Drop link-only or media-reference utterances
    if URL_RE.search(text):
        return False
    if text.lower() in ("[image]", "[video]", "[sticker]", "[gif]"):
        return False
    # Drop CoSEM placeholder tokens and media markers
    if "{{" in text or "}}" in text:
        return False
    if "<media omitted>" in text.lower():
        return False
    if "<file omitted>" in text.lower() or "<this message" in text.lower():
        return False
    # Drop utterances that are mostly emoji/punctuation (no letters)
    letters = sum(1 for c in text if c.isalpha())
    if letters < 10:
        return False
    # Drop utterances that are mostly numbers/codes (e.g. "10.30 b1!")
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    if digit_ratio > 0.3:
        return False
    return True

print("Parsing Corpus Chunks...")
all_utterances = []
chunk_files = sorted(
    f for f in os.listdir(extract_to) if f.endswith(".txt")
)
for fname in chunk_files:
    with open(os.path.join(extract_to, fname), encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff") 
            m = TAG_RE.match(line)
            if not m:
                continue
            tag, text = m.group(1), m.group(2).strip()
            if not is_usable(text):
                continue
            meta = parse_tag(tag)
            all_utterances.append({
                "tag": tag,
                "text": text,
                "chunk_file": fname,
                **meta,
                "gold_label": "",   
                "probe_tag": "",  
            })

print(f"Total usable utterances after filtering: {len(all_utterances):,}")

# Step 4: Score each Utterance for Stratified Sampling
# Singlish Discourse Particles
PARTICLES = {
    "lah", "lahh", "lahhh", "leh", "lehh", "lor", "lorh", "meh", "mehh",
    "hor", "horh", "sia", "siol", "siah", "liao", "liaos", "liaoz", "liaoooo",
    "liddat", "liddet", "lidat", "alamak", "aiyoh", "aiyo", "aiyah", "wah",
    "walao", "walaoeh", "wahlao", "eh", "lah!", "leh!", "meh?",
}
# Singlish Sentiment-Bearing Vocabulary
SINGLISH_SENT = {
    "shiok", "jialat", "sian", "swee", "cheem", "cui", "siao", "atas",
    "kiasu", "kiasi", "boring", "bo liao", "boliao", "tahan", "song",
    "die die", "confirm plus chop", "steady", "power", "syiok", "boh",
    "chui", "garang",
}
# Standard English Sentiment Cues
EN_SENT = {
    "love", "hate", "terrible", "amazing", "horrible", "awful", "fantastic",
    "wonderful", "awesome", "annoying", "stupid", "brilliant", "perfect",
    "worst", "best", "great", "bad", "good", "fail", "epic",
}
# Token-extraction Regex
TOKEN_RE = re.compile(r"[a-zA-Z]+[!?']?")

def signal_score(text):
    """
    Compute a 'Singlish-and-sentiment-density' score for an utterance.
    Higher score = more likely to exercise the project's failure modes.

    Components:
      +2 per Singlish particle token
      +3 per Singlish sentiment-bearing word
      +1 per English sentiment cue (capped at 2)
      +1 if the utterance contains '!!' or '!!!' (intensifier punctuation)
      +1 per emoji (capped at 2)
      +1 if the utterance contains a question mark followed by an explanation
        (a heuristic for the 'X meh? Y' sarcasm/scepticism pattern)
    """
    lowered = text.lower()
    tokens = [t.rstrip("!?'") for t in TOKEN_RE.findall(lowered)]
    score = 0
    score += 2 * sum(1 for t in tokens if t in PARTICLES)
    score += 3 * sum(1 for t in tokens if t in SINGLISH_SENT)
    # Multi-word Singlish Phrases
    for phrase in ("bo liao", "die die", "confirm plus chop", "steady pom pi pi"):
        if phrase in lowered:
            score += 3
    score += min(2, sum(1 for t in tokens if t in EN_SENT))
    if "!!" in text:
        score += 1
    # Rough Emoji Counter (any non-ASCII char, capped at 2)
    emoji_count = sum(1 for c in text if ord(c) > 127 and not c.isalpha())
    score += min(2, emoji_count)
    if "meh?" in lowered or "leh?" in lowered:
        score += 2
    return score

# Score all Utterances
print("Scoring utterances for Singlish/sentiment density...")
for u in all_utterances:
    u["_score"] = signal_score(u["text"])

high_signal = [u for u in all_utterances if u["_score"] > 0]
zero_signal = [u for u in all_utterances if u["_score"] == 0]
print(f"  High-signal (score > 0):  {len(high_signal):>7,}")
print(f"  Zero-signal (score == 0): {len(zero_signal):>7,}")

# Step 4b: Stratified Sample
random.seed(RANDOM_SEED)

# Sort High-Signal by Score (Descending)
high_signal.sort(key=lambda u: (-u["_score"], random.random()))

n_high_pilot = int(PILOT_N * HIGH_SIGNAL_PROPORTION)         
n_low_pilot = PILOT_N - n_high_pilot                          
n_high_main = int(MAIN_POOL_N * HIGH_SIGNAL_PROPORTION)       
n_low_main = MAIN_POOL_N - n_high_main                        

pilot_high = high_signal[:n_high_pilot]
main_high = high_signal[n_high_pilot : n_high_pilot + n_high_main]

random.shuffle(zero_signal)
pilot_low = zero_signal[:n_low_pilot]
main_low = zero_signal[n_low_pilot : n_low_pilot + n_low_main]

pilot = pilot_high + pilot_low
main_pool = main_high + main_low

random.shuffle(pilot)
random.shuffle(main_pool)

for u in pilot + main_pool:
    u["_score_internal"] = u.pop("_score") 

# Step 5: Create CSVs
def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")

write_csv(pilot, "cosem_pilot_sample.csv")
write_csv(main_pool, "cosem_main_pool.csv")

# Step 6: Preview Pilot for Inspection
print("\n=== First 15 Pilot Utterances (Preview) ===")
for i, u in enumerate(pilot[:15], 1):
    print(f"\n[{i:>2}] (score={u['_score_internal']}; {u['speaker_age']}yo "
          f"{u['ethnicity']}/{u['gender']}, {u['year']})")
    print(f"     {u['text']}")

# Summary Statistics
score_dist = {}
for u in pilot:
    s = u["_score_internal"]
    score_dist[s] = score_dist.get(s, 0) + 1
print(f"\nPilot Score Distribution: {dict(sorted(score_dist.items()))}")

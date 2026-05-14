from flask import Flask, render_template, request
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ─── Sample Dataset ───────────────────────────────────────────────
DATASET = [
    # REAL NEWS (label = 0)
    ("Government announces new infrastructure budget for 2025", 0),
    ("Scientists discover new treatment for diabetes", 0),
    ("Stock market closes higher after Federal Reserve decision", 0),
    ("Prime Minister addresses parliament on economic policy", 0),
    ("New study links air pollution to respiratory disease", 0),
    ("Supreme Court rules on landmark education case", 0),
    ("NASA launches new satellite to study climate change", 0),
    ("Local hospital opens new cancer treatment center", 0),
    ("Election commission announces voting schedule for state polls", 0),
    ("Tech company reports quarterly earnings above expectations", 0),
    ("Government increases minimum wage by ten percent", 0),
    ("World Health Organization issues new health guidelines", 0),
    ("Reserve Bank cuts interest rates to boost economy", 0),
    ("Police arrest three suspects in robbery case", 0),
    ("University researchers publish findings on renewable energy", 0),
    ("Municipal corporation to repair roads before monsoon", 0),
    ("International trade agreement signed between two nations", 0),
    ("Sports ministry approves funds for athlete training programs", 0),
    ("New railway line inaugurated connecting two major cities", 0),
    ("Court orders company to pay compensation to workers", 0),

    # FAKE NEWS (label = 1)
    ("SHOCKING! You have won 1 crore lottery click here now", 1),
    ("Government secretly plans to ban all mobile phones next week", 1),
    ("Drinking hot water cures cancer doctors dont want you to know", 1),
    ("URGENT: Send this message to 10 friends or face bad luck", 1),
    ("Aliens found in Antarctica government hiding the truth", 1),
    ("FREE iPhone giveaway claim yours before stock runs out", 1),
    ("Scientists CONFIRM earth is actually flat NASA lying to us", 1),
    ("Celebrity dies in tragic accident family demands justice NOW", 1),
    ("Secret cure for diabetes big pharma hiding from public", 1),
    ("BREAKING: Famous actor arrested for massive fraud click here", 1),
    ("Win free cash rewards just share your bank details", 1),
    ("Government to deposit 5000 rupees in everyone account tomorrow", 1),
    ("This miracle drink burns fat overnight doctors hate this trick", 1),
    ("EXPOSED: Politicians stealing money from poor citizens secretly", 1),
    ("Forward this to save a child dying of cancer please urgent", 1),
    ("New WhatsApp update will charge you money delete now", 1),
    ("Hindu Muslim riots spreading across entire country stay safe", 1),
    ("Bill Gates microchipping people through COVID vaccine conspiracy", 1),
    ("Breaking news prime minister resigns after secret scandal", 1),
    ("Earn 50000 per day working from home no experience needed", 1),
]

# ─── Keyword blacklist for fast detection ─────────────────────────
SPAM_KEYWORDS = [
    "click here", "win free", "you have won", "lottery", "claim now",
    "limited offer", "earn money", "make money fast", "secret cure",
    "doctors hate", "government hiding", "share your bank", "urgent forward",
    "send this to", "miracle", "shocking truth", "exposed", "they dont want",
    "whatsapp gold", "free iphone", "earn per day", "no experience needed",
    "congratulations you", "selected winner"
]

# ─── Preprocessing ────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── Train Model ──────────────────────────────────────────────────
def train_model():
    texts, labels = zip(*DATASET)
    cleaned = [preprocess(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(cleaned)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, labels)
    return vectorizer, clf

vectorizer, model = train_model()

# ─── Prediction ───────────────────────────────────────────────────
def predict_news(text):
    text_lower = text.lower()
    for kw in SPAM_KEYWORDS:
        if kw in text_lower:
            return "FAKE NEWS", "Keyword Detected", "danger"

    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = round(max(proba) * 100, 1)

    if pred == 1:
        return "FAKE NEWS", f"ML Model ({confidence}% confidence)", "danger"
    else:
        return "REAL NEWS", f"ML Model ({confidence}% confidence)", "success"

# ─── Routes ───────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    method = None
    style = None
    user_input = ""
    if request.method == 'POST':
        user_input = request.form.get('message', '').strip()
        if user_input:
            result, method, style = predict_news(user_input)
    return render_template('index.html',
                           result=result,
                           method=method,
                           style=style,
                           user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)

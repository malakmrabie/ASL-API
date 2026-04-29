from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()

# =========================
# Load spaCy model once
# =========================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise Exception("Run: python -m spacy download en_core_web_sm")

# =========================
# Request body
# =========================
class InputText(BaseModel):
    sentence: str

# =========================
# (نفس الكود بتاعك بدون تغيير)
# =========================

PRONOUN_MAP = {
    "i": "ME","me": "ME","my": "MY","myself": "ME",
    "you": "YOU","your": "YOUR","yourself": "YOU","yourselves": "YOU",
    "he": "HE","him": "HIM","his": "HIS","himself": "HE",
    "she": "SHE","her": "HER","hers": "HERS","herself": "SHE",
    "they": "THEY","them": "THEM","their": "THEIR","themselves": "THEY",
    "we": "WE","us": "US","our": "OUR","ourselves": "WE",
    "it": "IT","its": "ITS"
}

REMOVE_LEMMAS = {
    "be","have","do","can","could","will","would",
    "shall","should","may","might","must"
}

REMOVE_TOKENS = {
    "a","an","the","that","for","with","as","of",
    "in","on","at","by","about","from","every"
}

WH_WORDS = {"what","where","when","who","why","how","which"}

TIME_WORDS = {
    "today","tomorrow","yesterday","now","later","soon","never","always",
    "sometimes","tonight","morning","afternoon","evening","night",
    "week","month","year","day","hour","minute",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday"
}

def english_to_asl_gloss(sentence: str) -> list:
    doc = nlp(sentence.strip())
    tokens = list(doc)

    is_question = sentence.strip().endswith("?")

    time_glosses, subject_glosses = [], []
    object_glosses, verb_glosses = [], []
    other_glosses = []

    wh_gloss = None

    has_negation = any(
        t.lower_ in {"not", "n't"}
        or t.dep_ == "neg"
        or (t.lower_ == "no" and t.dep_ == "det")
        for t in tokens
    )

    for tok in tokens:
        low = tok.lower_
        lemma = tok.lemma_.lower()

        if tok.is_punct:
            continue

        if low in WH_WORDS and is_question:
            wh_gloss = low.upper()
            continue

        if low in REMOVE_TOKENS or lemma in REMOVE_LEMMAS or low == "to":
            continue

        if low in {"not", "n't", "no"}:
            continue

        if low == "there" and tok.dep_ == "expl":
            continue

        if tok.ent_type_:
            gloss = tok.text.upper()
        elif low in PRONOUN_MAP:
            gloss = PRONOUN_MAP[low]
        else:
            gloss = tok.lemma_.upper()

        if low in TIME_WORDS:
            time_glosses.append(gloss)
        elif tok.dep_ in {"nsubj", "nsubjpass"}:
            subject_glosses.append(gloss)
        elif tok.dep_ in {"dobj", "pobj", "attr", "dative"}:
            object_glosses.append(gloss)
        elif tok.pos_ in {"VERB", "AUX"}:
            verb_glosses.append(gloss)
        else:
            other_glosses.append(gloss)

    result = []
    result.extend(time_glosses)
    result.extend(subject_glosses)
    result.extend(object_glosses)
    result.extend(verb_glosses)

    if has_negation:
        temp = []
        inserted = False
        for g in result:
            temp.append(g)
            if not inserted and g in verb_glosses:
                temp.append("NOT")
                inserted = True
        if not inserted:
            temp.append("NOT")
        result = temp

    result.extend(other_glosses)

    if wh_gloss:
        result.append(wh_gloss)
    elif is_question:
        result.append("Q")

    return result

# =========================
# API Endpoint
# =========================
@app.post("/convert")
def convert_text(data: InputText):
    gloss = english_to_asl_gloss(data.sentence)

    return {
        "input": data.sentence,
        "gloss_list": gloss,
        "gloss_text": " ".join(gloss)
    }
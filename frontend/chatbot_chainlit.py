"""
Chainlit FAQ chatbot with ONNX embeddings (Sentence-Transformers).

Fixes your 404:
- The ONNX model file in `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  is located under `onnx/model.onnx`, not at repo root.

Features:
- Loads FAQ JSONL once on startup
- Downloads tokenizer + ONNX model from HF Hub
- Computes and caches question embeddings once
- Fast cosine-similarity retrieval with optional (degree, program) filtering
- Basic input validation + domain gating + contact routing

Run:
  chainlit run path/to/this_file.py
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import chainlit as cl
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAQ_FILE = PROJECT_ROOT / "data" / "generated" / "faqs_hybrid__llama3.1_8b.jsonl"

SIM_THRESHOLD_ANSWER = 0.55
SIM_THRESHOLD_OOD = 0.35

CONTACTS = {
    ("Bachelor", "Informatik"): "bsc-informatik@hhu.de",
    ("Master", "Informatik"): "msc-informatik@hhu.de",
    ("Master", "AI & Data Science"): "msc-ai-ds@hhu.de",
}

DOMAIN_KEYWORDS = (
    "prüfung", "klausur", "modul", "modulprüfung", "zulassung", "anmeldung",
    "wiederholung", "versuch", "frist", "ects", "leistungspunkte", "lp",
    "prüfungsausschuss", "prüfungsordnung", "bachelor", "master",
    "bachelorarbeit", "masterarbeit", "studiengang", "note", "benotet", "anerkennung",
    "einschreibung", "immatrikulation", "rücktritt", "täuschung", "plagiat"
)

GERMAN_INTERROGATIVES = (
    "wie", "was", "wann", "wo", "warum", "wieso", "weshalb",
    "wer", "wen", "wem", "welche", "welcher", "welches",
    "unter welchen", "in welchen"
)

COMMON_VERBS = (
    "ist", "sind", "wird", "werden", "kann", "können", "darf", "dürfen",
    "muss", "müssen", "soll", "sollen", "gilt", "gelten",
    "brauche", "braucht", "benötige", "benötigt",
    "erhalte", "erhält", "bekomme", "bekommt",
    "gibt", "geben", "läuft", "funktioniert", "verlängern",
    "anmelden", "beantragen", "ablehnen", "wiederholen"
)

# HF repo + filenames
ONNX_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ONNX_MODEL_FILENAME = "onnx/model.onnx"  # <-- IMPORTANT (fixes your 404)
TOKENIZER_FILENAME = "tokenizer.json"

MAX_LENGTH = 256


# ----------------------------
# Data types
# ----------------------------
@dataclass(frozen=True)
class QA:
    question: str
    answer: str
    degree_level: str
    program: str


# ----------------------------
# ONNX Embedder
# ----------------------------
class OnnxEmbedder:
    def __init__(self, repo_id: str, model_filename: str, tokenizer_filename: str, max_length: int = 256):
        self.repo_id = repo_id
        self.model_filename = model_filename
        self.tokenizer_filename = tokenizer_filename
        self.max_length = max_length

        self.model_path = hf_hub_download(repo_id=self.repo_id, filename=self.model_filename)
        self.tok_path = hf_hub_download(repo_id=self.repo_id, filename=self.tokenizer_filename)

        self.tokenizer = Tokenizer.from_file(self.tok_path)
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

        self.input_names = {i.name for i in self.session.get_inputs()}
        self.has_token_type_ids = "token_type_ids" in self.input_names

        # Tokenizer settings
        self.pad_id = self._get_pad_id_fallback()

    def _get_pad_id_fallback(self) -> int:
        """
        Many tokenizers define a PAD token; some don't expose it nicely via tokenizers API.
        We'll use 0 as a safe fallback (works for MiniLM tokenizer).
        """
        try:
            pad_id = self.tokenizer.token_to_id("[PAD]")
            if pad_id is not None:
                return int(pad_id)
        except Exception:
            pass
        return 0

    def _tokenize(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # tokenizers: encoding.ids, encoding.attention_mask available in many configs,
        # but not guaranteed. We'll build attention_mask ourselves.
        input_ids = []
        attention_mask = []
        token_type_ids = []

        for t in texts:
            enc = self.tokenizer.encode(t)
            ids = enc.ids[: self.max_length]
            mask = [1] * len(ids)

            pad_len = self.max_length - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_id] * pad_len
                mask = mask + [0] * pad_len

            input_ids.append(ids)
            attention_mask.append(mask)
            token_type_ids.append([0] * self.max_length)

        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)

        if self.has_token_type_ids:
            token_type_ids = np.asarray(token_type_ids, dtype=np.int64)
            return input_ids, attention_mask, token_type_ids
        return input_ids, attention_mask, None

    @staticmethod
    def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        # last_hidden: [B, T, H], attention_mask: [B, T]
        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (last_hidden * mask).sum(axis=1)          # [B, H]
        counts = np.clip(mask.sum(axis=1), 1e-9, None)     # [B, 1]
        return summed / counts

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, 1e-9, None)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        input_ids, attention_mask, token_type_ids = self._tokenize(texts)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.has_token_type_ids and token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.session.run(None, inputs)
        last_hidden = outputs[0]  # [B, T, H]

        pooled = self._mean_pool(last_hidden, attention_mask)
        pooled = self._l2_normalize(pooled.astype(np.float32))
        return pooled


# ----------------------------
# Loading FAQ dataset
# ----------------------------
def load_faqs(path: Path) -> List[QA]:
    qas: List[QA] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            if not q or not a:
                continue
            qas.append(
                QA(
                    question=q,
                    answer=a,
                    degree_level=(obj.get("degree_level") or "Unknown").strip(),
                    program=(obj.get("program") or "Unknown").strip(),
                )
            )
    if not qas:
        raise RuntimeError(f"No valid QAs found in {path}")
    return qas


# ----------------------------
# Validation helpers
# ----------------------------
def is_gibberish(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    letters = sum(ch.isalpha() for ch in t)
    return letters < max(3, int(len(t) * 0.25))


def looks_like_valid_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) < 15:
        return False
    if len(t.split()) < 4:
        return False
    if any(t.startswith(w) for w in GERMAN_INTERROGATIVES):
        return True
    if any(v in t for v in COMMON_VERBS):
        return True
    return False


def looks_in_domain(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in DOMAIN_KEYWORDS)


# ----------------------------
# Contact routing
# ----------------------------
def contact_suggestions_from_text(user_text: str) -> List[Tuple[str, str, str]]:
    t = (user_text or "").lower()

    mentions_bachelor = ("bachelor" in t) or ("b.sc" in t) or ("bsc" in t)
    mentions_master = ("master" in t) or ("m.sc" in t) or ("msc" in t)

    mentions_informatik = ("informatik" in t) or ("computer science" in t)
    mentions_ai_ds = ("data science" in t) or ("ai & data science" in t) or (("ai" in t) and ("data" in t))

    if mentions_bachelor:
        return [("Bachelor", "Informatik", CONTACTS[("Bachelor", "Informatik")])]
    if mentions_master and mentions_informatik:
        return [("Master", "Informatik", CONTACTS[("Master", "Informatik")])]
    if mentions_master and mentions_ai_ds:
        return [("Master", "AI & Data Science", CONTACTS[("Master", "AI & Data Science")])]

    # fallback: show all
    return [
        ("Bachelor", "Informatik", CONTACTS[("Bachelor", "Informatik")]),
        ("Master", "Informatik", CONTACTS[("Master", "Informatik")]),
        ("Master", "AI & Data Science", CONTACTS[("Master", "AI & Data Science")]),
    ]


def format_contacts_de(rows: List[Tuple[str, str, str]]) -> str:
    if len(rows) == 1:
        _, _, email = rows[0]
        return f"Bitte wende dich an: **{email}**"
    lines = "\n".join([f"- **{deg} {prog}**: {email}" for deg, prog, email in rows])
    return "Bitte wende dich an die passende Stelle:\n" + lines


# ----------------------------
# Retrieval
# ----------------------------
@dataclass
class Retriever:
    qas: List[QA]
    embeddings: np.ndarray  # [N, H] float32 normalized
    embedder: OnnxEmbedder

    def retrieve(self, query: str, degree: str, program: str) -> Tuple[int, float]:
        # filter indices by metadata (if user chose something other than Unknown)
        idxs = [
            i
            for i, qa in enumerate(self.qas)
            if (qa.degree_level == degree or degree == "Unknown")
            and (qa.program == program or program == "Unknown")
        ]
        if not idxs:
            idxs = list(range(len(self.qas)))

        qvec = self.embedder.encode([query])[0]     # [H]
        mat = self.embeddings[idxs]                 # [K, H]
        sims = mat @ qvec                           # cosine because normalized

        best_local = int(np.argmax(sims))
        best_idx = idxs[best_local]
        return best_idx, float(sims[best_local])


# ----------------------------
# Global state (loaded at startup)
# ----------------------------
RETRIEVER: Optional[Retriever] = None


def init_retriever() -> Retriever:
    qas = load_faqs(FAQ_FILE)

    embedder = OnnxEmbedder(
        repo_id=ONNX_REPO,
        model_filename=ONNX_MODEL_FILENAME,
        tokenizer_filename=TOKENIZER_FILENAME,
        max_length=MAX_LENGTH,
    )

    questions = [qa.question for qa in qas]
    embs = embedder.encode(questions).astype(np.float32)  # [N, H], normalized
    return Retriever(qas=qas, embeddings=embs, embedder=embedder)


# ----------------------------
# Chainlit UI
# ----------------------------
@cl.on_chat_start
async def start():
    global RETRIEVER
    if RETRIEVER is None:
        # Load everything once per process
        RETRIEVER = init_retriever()

    await cl.ChatSettings(
        [
            cl.Select(id="degree", label="Abschluss", values=["Unknown", "Bachelor", "Master"], initial_index=0),
            cl.Select(id="program", label="Studiengang", values=["Unknown", "Informatik", "AI & Data Science"], initial_index=0),
        ]
    ).send()

    await cl.Message(
        content=(
            "👋 Willkommen!\n\n"
            "Dieser Chatbot beantwortet Fragen **nur** anhand einer festen FAQ-Datenbank "
            "(aus Prüfungsordnungen)."
        )
    ).send()


@cl.on_message
async def handle(msg: cl.Message):
    global RETRIEVER
    if RETRIEVER is None:
        RETRIEVER = init_retriever()

    user_text = (msg.content or "").strip()

    if is_gibberish(user_text) or not looks_like_valid_question(user_text):
        await cl.Message(
            content=(
                "⚠️ Bitte formuliere eine vollständige und konkrete Frage.\n\n"
                "Beispiel: „Wie viele Prüfungsversuche habe ich pro Modul?“"
            )
        ).send()
        return

    if not looks_in_domain(user_text):
        await cl.Message(
            content=(
                "⚠️ Diese Frage passt wahrscheinlich nicht zur Prüfungsordnung (Informatik).\n\n"
                "Bitte frage zu Prüfungen, Modulen, Fristen, Zulassung, Bachelor-/Masterarbeit usw."
            )
        ).send()
        return

    settings: Dict[str, str] = cl.user_session.get("chat_settings") or {}
    degree = settings.get("degree", "Unknown")
    program = settings.get("program", "Unknown")

    best_idx, score = RETRIEVER.retrieve(user_text, degree, program)
    qa = RETRIEVER.qas[best_idx]

    if score < SIM_THRESHOLD_OOD:
        contacts = contact_suggestions_from_text(user_text)
        await cl.Message(
            content=(
                "⚠️ Ich kann dazu keine passende Stelle in der FAQ-Datenbank finden.\n\n"
                "Bitte ergänze mehr Kontext (z.B. Bachelor/Master, Studiengang, Modul, Frist).\n\n"
                + format_contacts_de(contacts)
            )
        ).send()
        return

    if score < SIM_THRESHOLD_ANSWER:
        contacts = contact_suggestions_from_text(user_text)
        await cl.Message(
            content=("❌ Ich konnte keine passende Antwort in der FAQ-Datenbank finden.\n\n" + format_contacts_de(contacts))
        ).send()
        return

    await cl.Message(content=qa.answer).send()

    await cl.Message(
        content=(
            f"**Gefundene FAQ:** {qa.question}\n"
            f"**Abschluss:** {qa.degree_level}\n"
            f"**Studiengang:** {qa.program}\n"
            f"**Similarity:** {score:.3f}"
        ),
        author="System",
        collapse=True,
    ).send()

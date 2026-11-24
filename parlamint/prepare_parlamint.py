'''
Prepares a sentence-level sample of ParlaMint-UA 4.0 for persuasion annotation.

Steps: download (if needed) -> keep regular Ukrainian speeches -> clean -> split into
sentences -> split sittings into LLM / human pools -> sample sentences whose length
distribution matches the Polish training data -> save CSV.

Corpus: http://hdl.handle.net/11356/1859 (CC BY 4.0)
'''

import ast
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "parlamint" / "raw"
TXT_DIR = RAW_DIR / "ParlaMint-UA.txt"
ARCHIVE = RAW_DIR / "ParlaMint-UA.tgz"
ARCHIVE_URL = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1859/ParlaMint-UA.tgz?sequence=39&isAllowed=y"
PL_TRAIN = ROOT / "data" / "original" / "PL_train.csv"
OUTPUT_DIR = ROOT / "data" / "parlamint"
# for people: id, then the text in reading order (context -> sentence), metadata at the end
READING_COLUMNS = ["id", "context", "content_uk"]
# annotators put 1 in each technique column that applies; done = 1 once the row is checked,
# so an all-empty row with done = 1 means "no persuasion"
ANNOTATION_COLUMNS = ["done", "comment"]
STRATEGIES = ["Attack_on_Reputation", "Justification", "Distraction", "Simplification", "Call", "Manipulative_Wording"]

SEED = 42
N_LLM = 3000            # sentences for LLM annotation
N_GOLD = 150            # human gold test (annotated by two people)
N_HUMAN_TRAIN = 250     # human-annotated training sentences
HUMAN_SITTING_SHARE = 0.15  # share of sittings reserved for human annotation
MIN_WORDS, MAX_WORDS = 2, 60  # PL fragments: median 13, 99th percentile 54 words
PER_SITTING_CAP = 30    # candidates per sitting, keeps the sample spread over all sittings
CONTEXT_SENTENCES = 2   # previous sentences shown to the annotator, not labeled
LENGTH_BINS = [MIN_WORDS - 1, 4, 10, 20, 40, MAX_WORDS]

UPPER = "А-ЯІЇЄҐA-Z"
NOTE_AFTER_END_RE = re.compile(r"([.!?…]\s*\([^()]{1,80}\))\s+")  # "колеги! (Оплески) Далі" -> note stays with the sentence
SENT_END_RE = re.compile(rf"([.!?…])\s+(?=[«\"“{UPPER}0-9])")
# shortcut: fixed abbreviation list, switch to Stanza's tokenizer if splits look wrong
ABBREV_RE = re.compile(rf"(?:\b[{UPPER}]|\b(?:ст|п|пп|м|р|рр|т|ч|ім|проф|тис|млн|млрд|грн|див|напр|вул|обл|с|акад|д|тов))\.$")
RUSSIAN_RE = re.compile(r"[ыэъё]", re.IGNORECASE)


def download_corpus() -> None:
    if TXT_DIR.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / ".gitignore").write_text("*\n")  # 580 MB of raw data must not reach git
    if not ARCHIVE.exists():
        print("Downloading ParlaMint-UA (137 MB)...")
        urllib.request.urlretrieve(ARCHIVE_URL, ARCHIVE)
    print("Extracting plain-text version...")
    with tarfile.open(ARCHIVE) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("ParlaMint-UA.txt/")]
        tar.extractall(RAW_DIR, members=members, filter="data")


def load_speeches() -> pd.DataFrame:
    read = dict(sep="\t", dtype=str, quoting=3, keep_default_na=False)  # quoting=3: quotes are part of the text
    meta = pd.concat((pd.read_csv(f, **read) for f in TXT_DIR.glob("*/*-meta-en.tsv")), ignore_index=True)
    text = pd.concat(
        (pd.read_csv(f, header=None, names=["ID", "text"], **read) for f in TXT_DIR.glob("*/*[0-9].txt")),
        ignore_index=True,
    )
    speeches = text.merge(meta, on="ID", validate="one_to_one")
    print(f"Speeches: {len(speeches):,}")

    # chairperson = procedure, guests = translated, Russian/multilingual = other language
    speeches = speeches[(speeches["Speaker_role"] == "Regular") & (speeches["Lang"] == "Ukrainian")]
    print(f"Regular Ukrainian speeches: {len(speeches):,}")
    return speeches.reset_index(drop=True)


def clean(text: str) -> str:
    text = re.sub(r"\[\[\d{1,2}:\d{2}(?::\d{2})?\]\]", " ", text)  # timestamps
    text = re.sub(r"\[\[(.*?)\]\]", r"(\1)", text)                  # [[Оплески]] -> (Оплески), as in PL data
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = NOTE_AFTER_END_RE.sub(r"\1\n", text)
    text = SENT_END_RE.sub(r"\1\n", text)
    sentences = []
    for part in text.split("\n"):
        if sentences and ABBREV_RE.search(sentences[-1]):
            sentences[-1] += " " + part  # undo split after "ст." or an initial like "В."
        else:
            sentences.append(part)
    return [s.strip() for s in sentences if s.strip()]


def to_sentences(speeches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for speech in speeches.itertuples(index=False):
        sentences = split_sentences(clean(speech.text))
        for i, sentence in enumerate(sentences):
            rows.append({
                "id": f"{speech.ID}.s{i}",
                "sitting": speech.Text_ID,
                "date": speech.Date,
                "term": speech.Term,
                "subcorpus": speech.Subcorpus,
                "speaker_party": speech.Speaker_party_name,
                "party_status": speech.Party_status,
                "party_orientation": speech.Party_orientation,
                "speaker_gender": speech.Speaker_gender,
                "context": " ".join(sentences[max(0, i - CONTEXT_SENTENCES):i]),
                "content_uk": sentence,
            })
    sentences = pd.DataFrame(rows)
    sentences["n_words"] = sentences["content_uk"].str.split().str.len()
    print(f"Sentences: {len(sentences):,}")

    sentences = sentences[
        sentences["n_words"].between(MIN_WORDS, MAX_WORDS)
        & ~sentences["content_uk"].str.contains(RUSSIAN_RE)  # Russian quotes inside Ukrainian speeches
    ]
    # frequent formulas ("Дякую за увагу.") stay once, like single procedural fragments in PL data
    sentences = sentences.drop_duplicates("content_uk")
    print(f"After length / language / duplicate filters: {len(sentences):,}")
    return sentences.reset_index(drop=True)


def split_sittings(sentences: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # whole sittings go to one pool, so the gold test never shares a debate with training data
    sittings = sentences[["sitting", "term"]].drop_duplicates()
    human = sittings.groupby("term")["sitting"].apply(
        lambda s: s.sample(frac=HUMAN_SITTING_SHARE, random_state=int(rng.integers(1 << 31)))
    )
    sentences["pool"] = np.where(sentences["sitting"].isin(set(human)), "human", "llm")
    return sentences


def length_weights(candidates: pd.DataFrame) -> pd.Series:
    # weight = PL share of a length bin / candidate share, so the sample length mix matches PL
    pl_words = pd.read_csv(PL_TRAIN)["content"].str.split().str.len()
    pl_share = pd.cut(pl_words, LENGTH_BINS).value_counts(normalize=True)
    bins = pd.cut(candidates["n_words"], LENGTH_BINS)
    pool_share = bins.value_counts(normalize=True)
    return bins.map(pl_share / pool_share).astype(float)


def sample_pool(sentences: pd.DataFrame, pool: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    candidates = (
        sentences[sentences["pool"] == pool]
        .sample(frac=1, random_state=int(rng.integers(1 << 31)))
        .groupby("sitting")
        .head(PER_SITTING_CAP)
    )
    return candidates.sample(n, weights=length_weights(candidates), random_state=int(rng.integers(1 << 31)))


def compare_lengths(sample: pd.DataFrame) -> None:
    pl_words = pd.read_csv(PL_TRAIN)["content"].str.split().str.len()
    table = pd.DataFrame({
        "PL_train": pd.cut(pl_words, LENGTH_BINS).value_counts(normalize=True),
        "ParlaMint_sample": pd.cut(sample["n_words"], LENGTH_BINS).value_counts(normalize=True),
    }).sort_index().round(3)
    print("\nLength distribution (words):")
    print(table.to_string())


def technique_strategy() -> dict[str, str]:
    # every technique belongs to exactly one strategy in PL data, read it from single-technique rows
    pl = pd.read_csv(PL_TRAIN)
    pairs = {
        (t[0], s[0])
        for t, s in zip(pl["techniques"].map(ast.literal_eval), pl["persuasion_strategies"].map(ast.literal_eval))
        if len(t) == 1 and len(s) == 1
    }
    mapping = dict(pairs)
    assert len(mapping) == len(pairs), "a technique maps to several strategies"
    return mapping


def technique_columns() -> list[str]:
    # grouped by strategy so related techniques stand next to each other for annotators
    mapping = technique_strategy()
    return sorted(mapping, key=lambda t: (STRATEGIES.index(mapping[t]), t))


def has_annotations(path: Path, label_columns: list[str]) -> bool:
    if not path.exists():
        return False
    existing = pd.read_csv(path, dtype=str)
    filled = [c for c in label_columns if c in existing.columns]
    return bool(filled) and existing[filled].notna().any().any()


def self_test() -> None:
    assert clean("Вітаю! [[Оплески]] [[12:32:41]] Далі") == "Вітаю! (Оплески) Далі"
    assert split_sentences("Вітаю, колеги! (Оплески) Нагадую, що ми працюємо. Згідно зі ст. 5 Закону все ясно.") == [
        "Вітаю, колеги! (Оплески)",
        "Нагадую, що ми працюємо.",
        "Згідно зі ст. 5 Закону все ясно.",
    ]
    assert split_sentences("Слово має В. Пашинський. Прошу.") == ["Слово має В. Пашинський.", "Прошу."]


def main() -> None:
    self_test()
    rng = np.random.default_rng(SEED)

    download_corpus()
    sentences = split_sittings(to_sentences(load_speeches()), rng)

    llm = sample_pool(sentences, "llm", N_LLM, rng).assign(split="llm_train")
    human = sample_pool(sentences, "human", N_GOLD + N_HUMAN_TRAIN, rng)
    # whole sittings go to gold until it has N_GOLD sentences, so gold may be slightly larger
    sitting_order = pd.Series(rng.permutation(human["sitting"].unique()))
    human = human.set_index("sitting").loc[sitting_order].reset_index()
    first_row_of_sitting = human.groupby("sitting", sort=False).cumcount() == 0
    gold_sittings = human.loc[first_row_of_sitting & (human.index < N_GOLD), "sitting"]
    human["split"] = np.where(human["sitting"].isin(gold_sittings), "gold_test", "human_train")

    sample = pd.concat([llm, human]).drop(columns="pool").sample(frac=1, random_state=SEED)
    assert sample["id"].is_unique and sample["content_uk"].is_unique
    sittings = sample.groupby("split")["sitting"].apply(set)
    assert sittings["gold_test"].isdisjoint(sittings["llm_train"] | sittings["human_train"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_DIR / "parlamint_ua_sample.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(sample):,} sentences to data/parlamint/parlamint_ua_sample.csv")
    label_columns = technique_columns() + ANNOTATION_COLUMNS
    for split, part in sample.groupby("split"):
        path = OUTPUT_DIR / f"parlamint_ua_{split}.csv"
        if split != "llm_train":
            if has_annotations(path, label_columns):
                print(f"{path.relative_to(ROOT)} already has annotations - not overwriting")
                continue
            # sorted by date and speech so annotators read debates in order; rows and values stay the same
            part = part.sort_values(["date", "id"])
            meta = [c for c in part.columns if c not in READING_COLUMNS]
            part = part[READING_COLUMNS].join(pd.DataFrame(index=part.index, columns=label_columns)).join(part[meta])
        part.to_csv(path, index=False, encoding="utf-8-sig")  # BOM so Excel shows Cyrillic correctly
        print(f"Saved {len(part):,} sentences to {path.relative_to(ROOT)}")

    print(f"Sittings covered: {sample['sitting'].nunique()}")
    compare_lengths(sample)


if __name__ == "__main__":
    main()

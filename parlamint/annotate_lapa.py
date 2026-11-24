'''
In Colab (GPU):
  !git clone <repo url> && cd <repo folder>
  !pip install -q -U transformers accelerate bitsandbytes
  from google.colab import drive; drive.mount("/content/drive")
  import os
  os.environ["HF_TOKEN"] = "..."
  os.environ["ANNOTATION_OUT_DIR"] = "/content/drive/MyDrive/lapa_annotation"
  !python parlamint/annotate_lapa.py --source pl_dev --n 50
  !python parlamint/annotate_lapa.py --source pl_test
  !python parlamint/annotate_lapa.py --source parlamint

'''

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from prepare_parlamint import STRATEGIES, technique_strategy


MODEL_ID = "lapa-llm/lapa-v0.1.3-instruct"
PROMPT_VERSION = "lapa-v1"
NEGATIVE_EXAMPLES = 10   # with one example per technique
MAX_NEW_TOKENS = 120     # the answer is a short JSON list
BATCH_SIZE = 4           # lower it if CUDA runs out of memory
JSON_RE = re.compile(r"\{.*?\}", re.S)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("ANNOTATION_OUT_DIR", ROOT / "data" / "parlamint" / "llm"))
PARLAMINT_FILE = ROOT / "data" / "parlamint" / "parlamint_ua_sample.csv"
PL_TRAIN_FILES = [ROOT / "data" / "translated" / "PL_train_translated_lapa.csv", ROOT / "data" / "original" / "PL_train.csv"]
PL_TEST_FILES = [ROOT / "data" / "translated" / "PL_test_translated_lapa.csv", ROOT / "PL_test_translated_lapa.csv"]
# pool of prompt examples that pl_dev excludes; same values as the Claude run, so pl_dev is the same rows
SEED = 42
POOL_EXAMPLES_PER_TECHNIQUE = 2
POOL_NEGATIVE_EXAMPLES = 20
N_DEV = 100

DEFINITIONS = {
    "Name_Calling-Labeling": "Labeling the target with a loaded word or phrase (insulting or praising) so the audience dislikes or likes them, instead of arguing.",
    "Guilt_by_Association": "Attacking the target by linking them to a person, group or idea the audience sees negatively.",
    "Doubt": "Undermining the credibility of a person, institution or claim by questioning their competence, honesty or results, often through insinuation or suggestive questions.",
    "Appeal_to_Hypocrisy": "Attacking the target by pointing out that their current words contradict their past actions or statements.",
    "Questioning_the_Reputation": "Attacking the moral character or reputation of the target with negative, often unsupported claims, rather than their argument.",

    "Flag_Waving": "Justifying a position by appealing to the pride or interests of a group (nation, party, religion, 'all Ukrainians').",
    "Appeal_to_Authority": "Supporting a claim mainly by the weight of an authority (expert, famous person, institution, the speaker themselves) rather than by evidence.",
    "Appeal_to_Popularity": "Claiming something is right because many people or the majority support it or do it.",
    "Appeal_to_Values": "Linking a position to values the audience sees as positive (freedom, democracy, family, justice, peace, tradition).",
    "Appeal_to_Fear-Prejudice": "Promoting or rejecting an idea by exploiting fear of a (often exaggerated) threat or prejudice against a group.",

    "Straw_Man": "Replacing the opponent's position with a distorted, weaker version and attacking that instead.",
    "Red_Herring": "Introducing irrelevant material to move attention away from the issue under discussion.",
    "Whataboutism": "Deflecting criticism by pointing to the opponent's other wrongdoing ('and what about you...') without answering it.",
    "Appeal_to_Pity": "Evoking pity, compassion or sympathy (suffering, victims, personal stories) to win support instead of giving arguments.",

    "Causal_Oversimplification": "Naming a single cause for a complex problem, or treating sequence or correlation as causation.",
    "Consequential_Oversimplification": "Claiming that a step will inevitably lead to a chain of (usually extreme) consequences without justification (slippery slope).",
    "False_Dilemma-No_Choice": "Presenting only two options, or only one, when more exist.",
    "False_Equivalence": "Presenting two things as equivalent or comparable when they differ in important ways.",

    "Slogans": "A short, catchy phrase used as a rallying cry or call to action.",
    "Conversation_Killer": "Words that discourage critical thought and close the discussion ('enough talk', 'that is how it is', thought-terminating clichés).",
    "Appeal_to_Time": "Arguing from urgency: 'now is the time', 'we cannot wait', 'it will be too late'.",

    "Loaded_Language": "Words or phrases with strong emotional connotations (positive or negative) used to influence the audience.",
    "Obfuscation-Vagueness-Confusion": "Deliberately vague, unclear or ambiguous wording that lets the audience read in their own meaning or avoids commitment.",
    "Exaggeration-Minimisation": "Representing something as much better, worse, bigger or smaller than it is, or downplaying its importance.",
    "Repetition": "Repeating the same word, phrase or idea to make it stick.",
}
TECHNIQUE_STRATEGY = technique_strategy()
TECHNIQUES = sorted(TECHNIQUE_STRATEGY)
assert set(DEFINITIONS) == set(TECHNIQUES), "definitions must match the PL taxonomy"


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    sys.exit(f"File not found, run `git pull` first: {paths[0].relative_to(ROOT)}")


def read_pl(path: Path) -> pd.DataFrame:
    pl = pd.read_csv(path)
    text_column = "content_uk" if "content_uk" in pl.columns else "content"
    return pd.DataFrame({
        "id": pl["filename"] + ":" + pl["start"].astype(str) + "-" + pl["end"].astype(str),
        "context": "",
        "content_uk": pl[text_column].fillna("").astype(str),
        "gold_techniques": pl["techniques"].map(ast.literal_eval),
    })


def example_pool(train: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    chosen = []
    for technique in TECHNIQUES:
        has = train[train["gold_techniques"].map(lambda labels: technique in labels) & ~train["id"].isin(chosen)]
        single = has[has["gold_techniques"].map(len) == 1]
        pool = single if len(single) >= POOL_EXAMPLES_PER_TECHNIQUE else has
        chosen += pool.sample(POOL_EXAMPLES_PER_TECHNIQUE, random_state=int(rng.integers(1 << 31)))["id"].tolist()
    negatives = train[(train["gold_techniques"].map(len) == 0) & ~train["id"].isin(chosen)]
    chosen += negatives.sample(POOL_NEGATIVE_EXAMPLES, random_state=SEED)["id"].tolist()
    return train.set_index("id").loc[chosen].sample(frac=1, random_state=SEED).reset_index()


def load_source(source: str, pool: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    if source == "parlamint":
        rows = pd.read_csv(PARLAMINT_FILE)
        return rows.assign(context=rows["context"].fillna(""))
    if source == "pl_dev":
        return train[~train["id"].isin(pool["id"])].sample(N_DEV, random_state=SEED).reset_index(drop=True)
    if source == "pl_test":
        return read_pl(first_existing(PL_TEST_FILES))
    sys.exit(f"Unknown source: {source}")


def definitions_text() -> str:
    return "\n".join(
        f"- {t} ({TECHNIQUE_STRATEGY[t]}): {DEFINITIONS[t]}"
        for t in sorted(TECHNIQUES, key=lambda t: (STRATEGIES.index(TECHNIQUE_STRATEGY[t]), t))
    )


def user_message(row) -> str:
    context = row.context.strip() or "(none, this is the first sentence of the speech)"
    return f"Context (preceding sentences, do not label):\n{context}\n\nTarget sentence:\n{row.content_uk}"


def lapa_examples(examples: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for technique in TECHNIQUES:
        has = examples[examples["gold_techniques"].map(lambda labels: technique in labels) & ~examples["id"].isin(chosen)]
        chosen += has["id"].head(1).tolist()
    chosen += examples.loc[examples["gold_techniques"].map(len) == 0, "id"].head(NEGATIVE_EXAMPLES).tolist()
    return examples[examples["id"].isin(chosen)]


def build_prompt(examples: pd.DataFrame) -> str:
    shots = "\n\n".join(
        f'Sentence: {row.content_uk}\nAnswer: {json.dumps({"techniques": list(row.gold_techniques)})}'
        for row in examples.itertuples()
    )
    return f"""You annotate persuasion techniques in speeches from the Ukrainian parliament (Verkhovna Rada).
List every technique from the taxonomy below that is present in the target sentence.
- Label only the target sentence. The context only helps to understand it; do not label techniques that occur only in the context.
- A sentence can contain several techniques, or none. Many parliamentary sentences are neutral or procedural.

Taxonomy (technique (strategy): definition):
{definitions_text()}

Annotated examples from Polish parliamentary speeches translated into Ukrainian:

{shots}

Answer with JSON only, without any other text: {{"techniques": [...]}}
Use only technique names from the taxonomy, spelled exactly as written. Use {{"techniques": []}} when there is none."""


def parse(text: str) -> dict:
    match = JSON_RE.search(text)
    try:
        techniques = json.loads(match.group())["techniques"] if match else None
    except (json.JSONDecodeError, KeyError, TypeError):
        techniques = None
    if not isinstance(techniques, list):
        return {"llm_error": "unparsable"}
    valid = sorted({t for t in techniques if t in TECHNIQUE_STRATEGY})
    return {"techniques": valid, "llm_explanation": "", "unknown": sorted({str(t) for t in techniques} - set(valid))}


def to_label_row(parsed: dict) -> dict:
    techniques = parsed.get("techniques")
    if techniques is None:
        return {"techniques": None, "persuasion_strategies": None, "llm_explanation": None, "llm_error": parsed["llm_error"]}
    strategies = sorted({TECHNIQUE_STRATEGY[t] for t in techniques}, key=STRATEGIES.index)
    return {
        "techniques": str(techniques),
        "persuasion_strategies": str(strategies),
        **{s: s in strategies for s in STRATEGIES},
        "llm_explanation": parsed["llm_explanation"],
        "llm_error": None,
    }


def write_labels(run: str, rows: pd.DataFrame, parsed: list[dict], meta: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame([to_label_row(p) for p in parsed], index=rows.index)
    out = rows.drop(columns=[c for c in labels.columns if c in rows.columns]).join(labels)
    out = out.assign(label_source="llm", **meta)
    path = OUT_DIR / f"{run}_labels.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} rows to {path} ({out['llm_error'].notna().sum()} errors)")
    return path


def evaluate(run: str) -> None:
    labels = pd.read_csv(OUT_DIR / f"{run}_labels.csv")
    if "gold_techniques" not in labels.columns:
        sys.exit("This run has no human labels to compare with.")
    labels = labels[labels["llm_error"].isna()]
    gold = labels["gold_techniques"].map(ast.literal_eval)
    pred = labels["techniques"].map(ast.literal_eval)
    print(f"Rows compared: {len(labels)}")

    print(f"\nDetection F1 (any technique): {f1_score(gold.map(bool), pred.map(bool)):.3f}")
    print(f"Positive share: gold {gold.map(bool).mean():.2f}, llm {pred.map(bool).mean():.2f}")
    for name, classes, to_set in (
        ("Strategies", STRATEGIES, lambda ts: {TECHNIQUE_STRATEGY[t] for t in ts}),
        ("Techniques", TECHNIQUES, set),
    ):
        binarizer = MultiLabelBinarizer(classes=classes)
        y_true, y_pred = binarizer.fit_transform(gold.map(to_set)), binarizer.transform(pred.map(to_set))
        present = np.flatnonzero(y_true.sum(axis=0))  # macro F1 only over classes present in gold
        print(f"\n{name}: micro F1 {f1_score(y_true, y_pred, average='micro', zero_division=0):.3f}, "
              f"macro F1 {f1_score(y_true, y_pred, average='macro', labels=present, zero_division=0):.3f}")
        per_class = pd.Series(f1_score(y_true, y_pred, average=None, zero_division=0), index=classes)
        support = pd.Series(y_true.sum(axis=0), index=classes)
        print(pd.DataFrame({"f1": per_class.round(3), "gold_count": support}).to_string())


def load_model():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    token = os.environ.get("HF_TOKEN")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quantization, device_map="auto", token=token
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    processor.tokenizer.padding_side = "left"  # batched generation needs left padding
    return model, processor


def generate(model, processor, prompts: list[str]) -> list[str]:
    import torch

    conversations = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]
    inputs = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", padding=True,
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)  # greedy: reproducible
    return processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="pl_dev", help="pl_dev | pl_test | parlamint")
    parser.add_argument("--n", type=int, help="annotate only the first n rows")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train = read_pl(first_existing(PL_TRAIN_FILES))
    pool = example_pool(train)
    rows = load_source(args.source, pool, train)
    rows = rows.head(args.n) if args.n else rows
    prompt = build_prompt(lapa_examples(pool))

    run = f"lapa_{args.source}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"{run}_raw.jsonl"
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            done[item["id"]] = item
    todo = rows[~rows["id"].isin(done)]
    print(f"{run}: {len(rows) - len(todo)} done earlier, {len(todo)} to annotate")

    if len(todo):
        model, processor = load_model()
        for start in range(0, len(todo), args.batch_size):
            chunk = todo.iloc[start:start + args.batch_size]
            texts = [f"{prompt}\n\n{user_message(row)}\n\nAnswer:" for row in chunk.itertuples()]
            outputs = generate(model, processor, texts)
            with raw_path.open("a", encoding="utf-8") as f:  # saved per batch: a disconnect loses at most one batch
                for row, output in zip(chunk.itertuples(), outputs):
                    item = {"id": row.id, "output": output, "parsed": parse(output)}
                    done[row.id] = item
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"{min(start + args.batch_size, len(todo))}/{len(todo)}  last answer: {outputs[-1].strip()[:80]}")

    parsed = [done[row_id]["parsed"] for row_id in rows["id"]]
    unknown = sorted({name for p in parsed for name in p.get("unknown", [])})
    if unknown:
        print(f"Dropped technique names that are not in the taxonomy: {unknown}")
    write_labels(run, rows, parsed, {"model": MODEL_ID, "prompt_version": PROMPT_VERSION})
    if "gold_techniques" in rows.columns:
        evaluate(run)


if __name__ == "__main__":
    main()

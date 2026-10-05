'''
This script works properly only if you have enough huggingface quota for GPU
Alternatively, I have built translation pipeline in Google Colab:
https://colab.research.google.com/drive/1ndHFaMMDYlUx50Gs2qEMcMOstbd4dxmi?usp=sharing
'''

import os
from pathlib import Path

import pandas as pd
from gradio_client import Client


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
ORIGINAL_DIR = DATA_DIR / "original"
TRANSLATED_DIR = DATA_DIR / "translated"

TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = ORIGINAL_DIR / "PL_train.csv"
OUTPUT_FILE = TRANSLATED_DIR / "PL_train_first_100_uk.csv"


def translate_text(client: Client, text: str) -> str:
    prompt = (
        "Translate the following Polish text into Ukrainian.\n"
        "Return only the Ukrainian translation. "
        "Preserve the meaning, names, numbers, quotations and tone.\n\n"
        f"Polish text:\n{text}"
    )

    history = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    result = client.predict(
        history,
        api_name="/bot",
    )

    return result[-1]["content"].replace("<end_of_turn>", "").strip()


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    first_100 = df.head(100).copy()

    client = Client(
    "lapa-llm/lapa",
    token=os.environ["HF_TOKEN"],
    )

    translations = []

    for i, text in enumerate(first_100["content"], start=1):
        print(f"Translating {i}/100...")

        translation = translate_text(client, str(text))
        translations.append(translation)

        print(translation)
        print("-" * 80)

    first_100["content_uk"] = translations

    first_100.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved translations to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
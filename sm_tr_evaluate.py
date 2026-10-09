import os
import pandas as pd
from comet import download_model, load_from_checkpoint

# Expects an environment variable named HF_TOKEN
# To set it in your terminal, run: export HF_TOKEN="your_hf_token_here"
hf_token = os.getenv("HF_TOKEN")

def main():
    print("Loading data...")
    df_lapa = pd.read_csv("PL_train_first_100_uk.csv")
    df_opus = pd.read_csv("PL_train_first_100_uk_opus.csv")
    # 1. Load the DeepL dataset
    df_deepl = pd.read_csv("PL_train_first_100_uk_deepl.csv") 

    # 2. Format for COMET-Kiwi
    data_lapa = [{"src": row["content"], "mt": str(row["content_uk"])} for _, row in df_lapa.iterrows()]
    data_opus = [{"src": row["content"], "mt": str(row["content_uk"])} for _, row in df_opus.iterrows()]
    data_deepl = [{"src": row["content"], "mt": str(row["content_uk"])} for _, row in df_deepl.iterrows()]

    print("Downloading/Loading COMET-Kiwi model...")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    # 3. Predict scores
    print("Evaluating Lapa...")
    lapa_output = model.predict(data_lapa, batch_size=8, gpus=0, num_workers=2)
    
    print("Evaluating Opus-MT...")
    opus_output = model.predict(data_opus, batch_size=8, gpus=0, num_workers=2)

    print("Evaluating DeepL...")
    deepl_output = model.predict(data_deepl, batch_size=8, gpus=0, num_workers=2)

    # 4. Print results
    print("\n--- COMET-KIWI RESULTS ---")
    print(f"Lapa Average Score:     {lapa_output.system_score:.4f}")
    print(f"Opus-MT Average Score:  {opus_output.system_score:.4f}")
    print(f"DeepL Average Score:    {deepl_output.system_score:.4f}")

    # 5. Save the output
    df_lapa['comet_score'] = lapa_output.scores
    df_opus['comet_score'] = opus_output.scores
    df_deepl['comet_score'] = deepl_output.scores
    
    df_lapa.to_csv("Lapa_Scored.csv", index=False)
    df_opus.to_csv("Opus_Scored.csv", index=False)
    df_deepl.to_csv("DeepL_Scored.csv", index=False)

if __name__ == "__main__":
    main()
import pandas as pd

df_lapa = pd.read_csv("Lapa_Scored.csv")
df_opus = pd.read_csv("Opus_Scored.csv")
df_deepl = pd.read_csv("DeepL_Scored.csv")

# Finding the biggest gaps between Lapa and Opus, while displaying DeepL as a benchmark
diff = df_lapa["comet_score"] - df_opus["comet_score"]
top_gaps_idx = diff.nlargest(5).index

for idx in top_gaps_idx:
    print("=" * 60)
    print(f"Index {idx} | Lapa Score: {df_lapa.loc[idx, 'comet_score']:.3f} | Opus Score: {df_opus.loc[idx, 'comet_score']:.3f} | DeepL Score: {df_deepl.loc[idx, 'comet_score']:.3f}")
    print(f"Original (PL): {df_lapa.loc[idx, 'content']}")
    print(f"Techniques:    {df_lapa.loc[idx, 'techniques']}")
    print(f"Lapa (UK):     {df_lapa.loc[idx, 'content_uk']}")
    print(f"Opus (UK):     {df_opus.loc[idx, 'content_uk']}")
    print(f"DeepL (UK):    {df_deepl.loc[idx, 'content_uk']}")
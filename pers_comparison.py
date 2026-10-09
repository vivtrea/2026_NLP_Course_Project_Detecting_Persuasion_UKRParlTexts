import pandas as pd
import ast

def main():
    print("Loading scored datasets...")
    # Load all three files
    df_lapa = pd.read_csv("Lapa_Scored.csv")
    df_opus = pd.read_csv("Opus_Scored.csv")
    df_deepl = pd.read_csv("DeepL_Scored.csv")

    # Rename columns for easy merging
    df_lapa = df_lapa[['content', 'techniques', 'content_uk', 'comet_score']].rename(
        columns={'content_uk': 'Lapa_UK', 'comet_score': 'Lapa_Score'}
    )
    df_opus = df_opus[['content', 'content_uk', 'comet_score']].rename(
        columns={'content_uk': 'Opus_UK', 'comet_score': 'Opus_Score'}
    )
    df_deepl = df_deepl[['content', 'content_uk', 'comet_score']].rename(
        columns={'content_uk': 'DeepL_UK', 'comet_score': 'DeepL_Score'}
    )

    # Merge them all based on the original Polish text
    df_merged = df_lapa.merge(df_opus, on='content').merge(df_deepl, on='content')

    # Convert stringified lists back to actual Python lists
    df_merged['techniques_list'] = df_merged['techniques'].apply(ast.literal_eval)

    # Filter out sentences that have NO persuasion techniques
    df_persuasion = df_merged[df_merged['techniques_list'].map(len) > 0].copy()

    # Define techniques that are notoriously hard to translate accurately
    hard_techniques = [
        'Loaded_Language', 'Appeal_to_Hypocrisy', 'Exaggeration-Minimisation', 
        'Name_Calling_Labeling', 'Doubt', 'Questioning_the_Reputation', 'Sarcasm'
    ]

    # Score each sentence based on how many "hard" techniques it contains
    df_persuasion['hard_score'] = df_persuasion['techniques_list'].apply(
        lambda x: sum(1 for tech in x if tech in hard_techniques)
    )

    # Sort by the hardest sentences first, and take the top 20
    # (If there are ties, drop_duplicates on the content ensures we get distinct sentences)
    top_20 = df_persuasion.sort_values(by='hard_score', ascending=False).drop_duplicates(subset=['content']).head(20)

    print(f"Extracted {len(top_20)} complex persuasion sentences. Generating Markdown report...")

    # Write the comparison to a Markdown file for easy reading
    with open("Persuasion_Comparison.md", "w", encoding="utf-8") as f:
        f.write("# Translation Comparison for Persuasion Techniques\n\n")
        f.write("Review these sentences to see which model best preserves the emotional and rhetorical tone.\n\n")
        
        for idx, row in top_20.iterrows():
            f.write(f"### Example {idx + 1}\n")
            f.write(f"**Techniques Detected:** `{row['techniques']}`\n\n")
            f.write(f"> **🇵🇱 Original (PL):** {row['content']}\n\n")
            
            f.write(f"**🤖 Lapa LLM** *(Score: {row['Lapa_Score']:.3f})*:\n")
            f.write(f"{row['Lapa_UK']}\n\n")
            
            f.write(f"**🌐 DeepL API** *(Score: {row['DeepL_Score']:.3f})*:\n")
            f.write(f"{row['DeepL_UK']}\n\n")
            
            f.write(f"**⚙️ Opus-MT** *(Score: {row['Opus_Score']:.3f})*:\n")
            f.write(f"{row['Opus_UK']}\n\n")
            f.write("---\n\n")
            
    print("Done! Open 'Persuasion_Comparison.md' in VS Code to review the translations.")

if __name__ == "__main__":
    main()
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import torch

def main():
    model_name = "Helsinki-NLP/opus-mt-pl-uk"
    print(f"Loading {model_name}...")
    
    # Load model and tokenizer directly
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print("Loading data...")
    df = pd.read_csv("PL_train_first_100_uk.csv")
    
    # Extract Polish sentences
    polish_texts = df["content"].tolist()
    
    print(f"Translating {len(polish_texts)} sentences...")
    start_time = time.time()
    
    translations = []
    batch_size = 16
    
    # Process in batches manually
    for i in range(0, len(polish_texts), batch_size):
        batch = polish_texts[i : i + batch_size]
        
        # 1. Convert text to tokens
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        
        # 2. Generate translation tokens (CPU is completely fine here)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
        
        # 3. Decode tokens back to readable Ukrainian text
        batch_translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translations.extend(batch_translations)
        
    print(f"Translation finished in {time.time() - start_time:.2f} seconds.")
    
    # Replace the translation column
    df["content_uk"] = translations
    
    output_file = "PL_train_first_100_uk_opus.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved translated data to {output_file}")

if __name__ == "__main__":
    main()
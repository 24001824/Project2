import pandas as pd
import random
import torch
from transformers import pipeline
from datasets import load_dataset

# Student Name: Obaid AlShamsi - Saeed AlSaalty
# ID: 24001824 - 24001920
# Reproduction of SSTuning (ACL 2023)

def main():
    print("Loading model...")
    # Using the official pre-trained model from the paper
    # device=-1 for CPU, change to 0 if using Colab GPU
    classifier = pipeline("zero-shot-classification", 
                          model="DAMO-NLP-SG/zero-shot-classify-SSTuning-base",
                          device=-1)

    print("Loading SST-2 validation set...")
    # Grab first 100 examples for the benchmark
    dataset = load_dataset("sst2", split="validation[:100]")

    results = []
    
    for i, row in enumerate(dataset):
        text = row['sentence']
        # Label 1 is positive, 0 is negative in SST-2
        gt_label = "positive" if row['label'] == 1 else "negative"
        candidates = ["positive", "negative"]

        # 1. Standard Zero-Shot Inference
        # The paper uses entailment, so we stick to the default template structure
        out = classifier(text, candidates, hypothesis_template="This text is about {}.")
        pred = out['labels'][0]
        
        # 2. Robustness Test (Adversarial Typo)
        # Just swap two chars to simulate noise
        if len(text) > 4:
            chars = list(text)
            idx = random.randint(0, len(text) - 2)
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            text_noisy = "".join(chars)
        else:
            text_noisy = text
            
        out_noisy = classifier(text_noisy, candidates, hypothesis_template="This text is about {}.")
        pred_noisy = out_noisy['labels'][0]

        # Track stats
        is_correct = (pred == gt_label)
        is_stable = (pred == pred_noisy)

        results.append({
            "text": text,
            "ground_truth": gt_label,
            "pred_clean": pred,
            "pred_noisy": pred_noisy,
            "correct": is_correct,
            "stable": is_stable
        })

        if i % 10 == 0:
            print(f"Processed {i}/100 samples...")

    # Stats
    df = pd.DataFrame(results)
    acc = df["correct"].mean() * 100
    stab = df["stable"].mean() * 100
    
    print("\n--- FINAL RESULTS (100 Samples) ---")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Robustness: {stab:.2f}%")
    
    df.to_csv("results.csv", index=False)
    print("Saved detailed logs to results.csv")

if __name__ == "__main__":
    main()
# NLP Assignment 2: Reproduction of SSTuning (ACL 2023)


## 📄 Paper Details
* **Title:** Zero-Shot Text Classification via Self-Supervised Tuning
* **Authors:** Chaoqun Liu et al. (DAMO Academy)
* **Venue:** Findings of ACL 2023
* **Original Repository:** [DAMO-NLP-SG/SSTuning](https://github.com/DAMO-NLP-SG/SSTuning)

## 🎯 Project Objective
The goal of this project is to reproduce the **Zero-Shot Generalization** capabilities of the SSTuning framework. The central claim of the paper is that their model can classify text into unseen categories without requiring task-specific fine-tuning.

To verify this, I performed an **Inference-Based Reproduction** using the authors' official pre-trained checkpoints on the SST-2 sentiment analysis benchmark.

## 📊 Methodology
1. **Model:** I utilized the official checkpoint `DAMO-NLP-SG/zero-shot-classify-SSTuning-base`.
2. **Inference Strategy:** I used the Entailment-based inference method described in the paper (converting labels into hypothesis templates like *"This text is about {}"*).
3. **Dataset:** I evaluated the model on a subset (N=100) of the **SST-2** (Stanford Sentiment Treebank) validation set.
4. **Original Contribution:** I added a **Robustness Test** to measure model stability. I introduced adversarial noise (random character swaps/typos) to the input text to see if the zero-shot capabilities degrade on noisy data.

## 📉 Key Findings

Running the reproduction script on 100 samples yielded the following results:

| Metric     | Result  | Interpretation                                                                 |
|------------|---------|--------------------------------------------------------------------------------|
| Accuracy   | 81.00%  | The model successfully performs zero-shot classification on the majority of samples, confirming the paper's claims of strong generalization. |
| Robustness | 91.00%  | The model is highly stable; in 91% of cases, introducing typos did not change the model's prediction, suggesting it relies on robust sub-word features. |

## 💻 How to Run the Code
This code is designed to run in **Google Colab** or any local Python environment.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
# Reading Between the Lines: Improving Sentiment Analysis in Singlish with Fine-Tuned Transformer Models 🇸🇬💬

<p align="center">
  <img src="singlish-banner.jpeg" alt="Singlish Sentiment">
</p>

## Abstract
Singlish (or ‘Singapore English’) is an informal, colloquial form of English that is commonly used in Singapore (Chi, 2025). It is increasingly gaining global recognition as a distinct and evolving creole language instead of being dismissed as mere ‘broken English’. However, Singlish remain severely under-represented in NLP research especially as it often contains sentiment-bearing discourse particles such as 'lah', 'lor' and 'meh' which does not have direct lexical equivalents in the Standard English copora.

## Table of Contents
1. [Introduction](#introduction)
2. [Repository Contents](#repository-contents)
3. [Conclusion](#conclusion)
4. [Future Developments](#future-developments)
6. [Acknowledgements](#acknowledgements)


## Introduction
In this project, the Corpus of Singapore English Messages is used as the primary data source. The main study (not yet conducted) will explore whether Fine-tuning a Singlish-pretrained transformer on a labelled Singlish sentiment corpus will
yield higher macro-F1 on Singlish texts than fine-tuning a larger multilingual transformer.
on the same data.

## Repository Contents

| File | Purpose |
|------|---------|
| `pilot_baselines.py` | Baseline Evaluation on the 18-item Probe Set |
| `cosem_acquire.py` | Stratified Sampling of CoSEM |
| `cosem_finetune_pilot.ipynb` | Fine-Tuning of SingBERT |
| `cosem_main_pool.csv` | The CoSEM utterances intended for the Main Study|
| `cosem_pilot_sample.csv` | The 100 CoSEM utterances with author-assigned labels for the Pilot Study |


## Conclusion

In this proposal, we set out the rationale, pilot evidence, and six-month plan for a Singlish sentiment-analysis project grounded in fine-tuned transformer models.

## Conclusion
- **CoSEM** — Corpus of Singapore English Messages (Gonzales et al., 2021)
- **SingBERT** — pre-trained Singlish BERT (Lim, 2020), available on HuggingFace

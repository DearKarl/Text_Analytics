# NLP for Climate-related Text Analytics

Latent topic discovery using LDA and sentence embeddings on ClimateBERT, and named entity recognition on Twitter using CRF baselines and a fine-tuned BERT model

## Abstract

This project develops two complementary natural language processing (NLP) pipelines for analysing climate-related communication across formal disclosures and social media.  

1) **Climate Topic Modelling** the **ClimateBERT** sentiment corpus is preprocessed and analysed using Latent Dirichlet Allocation (LDA) and embedding-based clustering (SentenceTransformers + K-Means). Topics and clusters are semantically classified as *risk* or *opportunity* via cosine similarity to anchor sentences, and results are visualised with pyLDAvis and UMAP.  

2) **Twitter NER:** feature-engineered Conditional Random Field (CRF) models and a fine-tuned BERT token classifier are applied to the Broad Twitter Corpus (BTC). Evaluation with **seqeval** (micro, macro, and weighted averages) is complemented by error-transition analysis, confusion heatmaps, and token-level comparisons.  

## Background  

The analysis of climate-related texts requires methods capable of handling both the specialised language of corporate reports and the noisy, informal style of social media. Traditional approaches such as LDA and CRF provide interpretable baselines for topic discovery and sequence labelling, but they are limited in capturing higher-level semantics and robustness on noisy data. Deep learning methods, including SentenceTransformers for embedding-based clustering and BERT for token classification, offer richer contextual representation and improved generalisation. This project combines these complementary approaches to examine how climate-related risks and opportunities are expressed across formal disclosures and user-generated content.  

## Project Goals  

1) **Climate Topic Modelling**  
  - Preprocess ClimateBERT sentiment texts for a binary risk–opportunity task.  
  - Apply LDA with Count and TF-IDF vectorisation (5/10 topics).  
  - Map topics to semantic classes via cosine similarity to anchor sentences.  
  - Cluster documents using transformer embeddings (MiniLM) with K-Means, and visualise with UMAP.  
  - Generate interactive pyLDAvis HTML outputs for exploration.  

2) **Twitter NER**  
  - Implement CRF baseline models.  
  - Fine-tune BERT for token-level classification with subword alignment.  
  - Evaluate using seqeval precision, recall, and F1 (micro, macro, weighted).  
  - Perform error analysis with transition statistics and token-level comparisons.  

## Methods Overview  

1) **ClimateBERT Topic Modelling**  

- **Preprocessing:** lowercasing, tokenisation, stopword removal, WordNet lemmatisation.  
- **LDA:** scikit-learn (5/10 topics) and Gensim (BoW and TF-IDF), visualised with pyLDAvis.  
- **Semantic classification:** SentenceTransformer embeddings compared to risk/opportunity anchors.  
- **Clustering:** MiniLM embeddings → K-Means (k=5) → semantic labelling by anchor similarity.  

2) **Twitter NER**  

- **Dataset:** Broad Twitter Corpus (BTC) from the TNER benchmark.  
- **Models:**  
  - CRF-1: lexical/orthographic features,  
  - CRF-2: + POS tags,  
  - CRF-3: + Twitter-specific cues,  
  - BERT: bert-base-cased fine-tuned with Hugging Face Trainer.  
- **Evaluation:** micro/macro/weighted precision, recall, F1; per-entity reports; confusion heatmaps.  
- **Error analysis:** transition error distributions, mismatch case studies, and side-by-side token comparisons.  

## Data Sources  

- **ClimateBERT Sentiment:** [Hugging Face – climatebert/climate_sentiment](https://huggingface.co/datasets/climatebert/climate_sentiment)  
- **Broad Twitter Corpus (BTC):** [Hugging Face – tner/btc](https://huggingface.co/datasets/tner/btc)  

## References  

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation.*  
- Sievert, C., & Shirley, K. (2014). *pyLDAvis: Interactive Topic Model Visualisation.*  
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection.*  
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT.*  
- Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional Random Fields.*  
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.*  

---

> Repository maintained by [DearKarl](https://github.com/DearKarl). Contributions and feedback welcome.  

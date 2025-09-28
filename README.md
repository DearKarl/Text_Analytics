# Climate-related Topic Modelling & Twitter NER
### ClimateBERT topic modelling with embeddings and Twitter named entity recognition using CRF and BERT

This repository implements two complementary NLP pipelines:  

1. **Climate-related topic modelling and semantic clustering** on the ClimateBERT sentiment dataset, combining traditional LDA with modern embedding-based methods to uncover themes of *climate risk* and *climate opportunity*.  
2. **Named Entity Recognition (NER) on Twitter text**, comparing three feature-engineered Conditional Random Field (CRF) baselines with a fine-tuned BERT token classification model, evaluated with entity-level metrics and error analysis.  


## Background  

Climate disclosures and social media data present distinct challenges: domain-specific terminology in financial and environmental reports, and noisy, informal language on Twitter.  

- **For climate texts**, word co-occurrence via Latent Dirichlet Allocation (LDA) provides interpretable topic structures but struggles to map directly to semantic classes like *risk* vs *opportunity*. Embedding-based clustering with SentenceTransformers addresses this by leveraging contextual meaning, and semantic anchors enable consistent topic labelling.  
- **For Twitter NER**, CRF models remain strong baselines when enhanced with lexical, POS, and Twitter-specific features, while transformer models (BERT) provide superior contextual understanding. By comparing both approaches, we reveal trade-offs in robustness and generalisation across noisy, user-generated text.  


## Project Goals  

- **Climate Topic Modelling**  
  - Preprocess ClimateBERT sentiment texts into a binary task (*risk* vs *opportunity*).  
  - Apply LDA with both Count and TF-IDF vectorisation (5/10 topics).  
  - Map topics to semantic classes via cosine similarity to anchor sentences.  
  - Cluster documents using transformer embeddings (MiniLM) + K-Means, and visualise with UMAP.  
  - Provide interactive pyLDAvis HTML outputs for exploration.  

- **Twitter NER**  
  - Implement three CRF baselines:  
    - CRF-1 (basic lexical features),  
    - CRF-2 (+ POS tags),  
    - CRF-3 (+ Twitter cues such as @mentions, #hashtags, URLs).  
  - Fine-tune BERT for token-level classification with subword alignment.  
  - Evaluate models with seqeval precision/recall/F1 (micro, macro, weighted averages).  
  - Analyse top error transitions and show token-level comparisons across models.  


## Methods Overview  

### A. ClimateBERT Topic Modelling  

- **Preprocessing**: lowercasing, tokenisation, stopword removal, WordNet lemmatisation.  
- **LDA**: sklearn (5/10 topics) and Gensim (BoW and TF-IDF), visualised with pyLDAvis.  
- **Semantic classification**: SentenceTransformer embeddings compared to *risk*/*opportunity* anchors.  
- **Clustering**: MiniLM embeddings → K-Means (k=5) → semantic labelling by anchor similarity.  
- **Visualisation**: UMAP 2D projection with cluster colouring.  

### B. Twitter NER  

- **Dataset**: Broad Twitter Corpus (BTC) from the TNER benchmark.  
- **Models**:  
  - CRF-1: lexical/orthographic features.  
  - CRF-2: + POS tags.  
  - CRF-3: + Twitter-specific binary cues.  
  - BERT: bert-base-cased fine-tuned with Hugging Face Trainer.  
- **Evaluation**: micro-averaged precision, recall, F1; per-entity reports; confusion heatmaps.  
- **Error analysis**: transition error distributions, mismatch case studies, and side-by-side token tables.  


## Data Sources  

- **ClimateBERT Sentiment**: [Hugging Face – climatebert/climate_sentiment](https://huggingface.co/datasets/climatebert/climate_sentiment)  
- **Broad Twitter Corpus (BTC)**: [Hugging Face – tner/btc](https://huggingface.co/datasets/tner/btc)  


## References  

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation.*  
- Sievert, C., & Shirley, K. (2014). *pyLDAvis: Interactive Topic Model Visualisation.*  
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection.*  
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT.*  
- Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional Random Fields.*  
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.*  

---

> Repository maintained by [DearKarl](https://github.com/DearKarl). Contributions and feedback welcome.  

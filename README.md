# 📑 Project Documentation: Image Captioning using Deep Learning

## 1. Introduction

Image Captioning is a challenging task at the intersection of **Computer Vision** and **Natural Language Processing**, where the goal is to generate meaningful textual descriptions for images. This project explores various architectures and techniques to improve caption generation performance, beginning with classical CNN-LSTM approaches and progressing toward Transformer-based models with attention.

---

## 2. Dataset

- **Flickr8k Dataset**: Initially used for experimentation due to its small size and faster training.
- **Flickr30k Dataset**: Adopted later to provide a larger and more diverse set of captions for improved generalization.

---

## 3. Experimental Progression

### 3.1 Baseline: CNN + LSTM (TensorFlow)

- **Backbone**: InceptionNet for feature extraction.
- **Decoder**: LSTM-based sequence generator.
- **Result**:

  - Model failed to generate coherent captions.
  - Very low validation accuracy (\~8%).

---

### 3.2 Hyperparameter Tuning Attempts

- Increased training epochs.
- Modified learning rates, optimizers, and batch sizes.
- **Observation**: No significant improvements.

---

### 3.3 EfficientNet Backbone

- Replaced InceptionNet with **EfficientNet**.
- Extracted features on-the-fly.
- **Result**: Slight improvement in sentence formation, but captions were still weak.

---

### 3.4 Adding Attention (Bahdanau Attention)

- Implemented **Bahdanau Attention** to focus on relevant image features during decoding.
- Used **teacher forcing** during training.
- **Result**: Validation accuracy improved from **8% → 15%**.
- Captions became more structured but still lacked relevance to images.

---

### 3.5 Transformer Decoder

- Replaced LSTM decoder with a **Transformer-based decoder** (inspired by _Attention is All You Need_).
- **Result**:

  - Validation accuracy increased to **40%**.
  - Captions were more coherent and grammatically correct.
  - However, generated captions often failed to describe the actual image content accurately.

---

### 3.6 Larger Dataset: Flickr30k

- Switched to Flickr30k for more diverse training samples.
- **Result**: Small performance boost, but captions were still not satisfactory in terms of semantic relevance.

---

### 3.7 CLIP ViT Feature Extractor

- Experimented with **OpenAI CLIP’s ViT** as a backbone for feature extraction.
- **Result**: No significant performance improvement.
- Model size and training complexity increased considerably.

---

## 4. Current Model

- **Backbone**: EfficientNet for image feature extraction.
- **Decoder**: Transformer Decoder.
- **Training Strategy**: Hyperparameter tuning focused on decoder architecture (e.g., number of layers, hidden dimensions, attention heads).
- **Performance**:

  - Validation accuracy \~40%.
  - Captions are grammatically correct but still struggle with semantic alignment to the input image.

---

## Flow

```mermaid
flowchart TD

    A[Start Project] --> B[Use Flickr8k Dataset]
    B --> C[InceptionNet + LSTM Architecture]
    C --> D[Performance very poor: captions not meaningful]

    D --> E[Hyperparameter tuning<br>(epochs, LR, optimizer, batch size)]
    E --> F[No major improvement]

    F --> G[Changed Backbone to EfficientNet<br>+ Feature Extraction on the fly]
    G --> H[Slight performance improvement]

    H --> I[Added Bahdanau Attention + Teacher Forcing]
    I --> J[Validation Accuracy ↑ from 8% → 15%]

    J --> K[Replaced LSTM Decoder with Transformer Decoder<br>(Inspired by Attention is All You Need)]
    K --> L[Validation Accuracy ↑ to 40%<br>Captions made sense but not strongly image-related]

    L --> M[Scaled Dataset → Flickr30k]
    M --> N[Minor improvement]

    N --> O[Tried CLIP ViT as feature extractor]
    O --> P[No improvement, model became heavier]

    P --> Q[Final Setup]
    Q --> R[EfficientNet Backbone + Transformer Decoder]
    R --> S[Currently experimenting with hyperparameters<br>especially Decoder size and parameters]

    style A fill:#3e8,stroke:#333,stroke-width:2px
    style Q fill:#6cf,stroke:#333,stroke-width:2px
    style S fill:#fc6,stroke:#333,stroke-width:2px
```

---

## 5. Challenges

- **Dataset Size**: Flickr datasets are relatively small compared to large-scale captioning datasets (e.g., MSCOCO).
- **Semantic Alignment**: While the model produces syntactically correct sentences, relevance to the input image remains a challenge.
- **Heavy Architectures**: Using CLIP increased computational cost without clear gains.

---

## 6. Future Work

- Explore **MSCOCO dataset** for better training diversity.
- Experiment with **pre-trained vision-language models** (e.g., BLIP, OFA, Flamingo).
- Incorporate **beam search or nucleus sampling** for better caption generation.
- Fine-tune CLIP features instead of using them as frozen embeddings.
- Add **reinforcement learning (CIDEr optimization)** to align captions with evaluation metrics.

---

## 7. Conclusion

This project highlights the progression from **basic CNN-LSTM models** to **Transformer-based architectures with attention** in image captioning. While performance improved significantly (from incoherent captions to 40% validation accuracy), the key challenge remains generating captions that **semantically align with the image content**. Ongoing work focuses on scaling datasets, refining decoders, and leveraging powerful pretrained vision-language models.

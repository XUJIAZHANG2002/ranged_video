# DIFFUSION MODELS FOR MULTI-MODAL GENERATIVE MODELING
https://arxiv.org/pdf/2407.17571
# Boosting Generative Image Modeling via Joint Image-Feature Synthesis
https://arxiv.org/html/2504.16064v1
# A Multimodal Dynamical Variational Autoencoder for Audiovisual Speech Representation Learning⋆
https://arxiv.org/pdf/2305.03582

# Unconditional Image-Text Pair Generation with Multimodal Cross Quantizer
https://arxiv.org/pdf/2204.07537

# Unified Multi-modal Image Generation and Understanding
https://arxiv.org/pdf/2503.20644v1


```
           ┌──────────────────┐
Semantic ─►│ Semantic Encoder │──┐
           └──────────────────┘  │
                                 ├─► concat ─► Quantizer (1 codebook) ─► Decoder backbone ─┬─► Sem Head
           ┌──────────────────┐  │                                                        └─► Depth Head
 Depth ───►│  Depth Encoder   │──┘
           └──────────────────┘
```
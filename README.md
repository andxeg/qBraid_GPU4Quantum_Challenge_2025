# JPMorgan-Challenge-Submission

**Single-shot circuit generation + graph-partitioning pipeline that delivers up to 137× speed-ups on 20-asset problems and scales cleanly to 150-asset instances.**

## ✨ Key ideas

- **GPT-based circuit generator** – swaps the slow, iterative QAOA optimiser for a single transformer inference, eliminating hundreds of gradient-descent steps.
- **Warm-start option** – feed the GPT suggestion into any classical optimiser when you want a final polish.
- **Hardware-aware graph partitioner** – decomposes a large portfolio into _p_-node sub-graphs that fit comfortably on your GPU/QPU, then stitches the sub-solutions back together.
- **Full pipeline notebook** – demonstrates end-to-end optimisation of a 150-asset Nasdaq portfolio, including RMT denoising, spectral clustering, and recombination of sub-solutions.

## Repository layout

```
.
├── gpt-qaoa/              # Inference engine + (optional) fine-tuning utilities
│   ├── model.py
│   ├── generate.py
│   └── train.py
├── partitioning/          # Graph decomposition à la Acharya et al.
│   ├── Partitioning_classical_approach_210 nodes.ipynb # demonstration of decomposition pipeline
│   ├── partitioning_with_gpt.py # GPT-based circuit generator in paritioning workflow (working progress)
│   ├── partitioning_with_QOKit.py # QOKit-based circuit simulator in partitioning workflow (working progress)
│   └── requirements.txt
├── scripts/               # CLI entry points & benchmark runners
│   ├── solve_small.py     # 5- to 20-node experiments
│   └── solve_large.py     # 150-asset pipeline
├── notebooks/             # Jupyter walkthroughs of every experiment
├── data/                  # Example graphs, covariance matrices, checkpoints*
├── requirements.txt
└── README.md              # (← you are here)
\* Large binary assets are stored via Git LFS.
```

]

```

```

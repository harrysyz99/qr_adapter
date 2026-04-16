<h1 align="center">
QR-Adaptor: Balancing Fidelity and Plasticity for Mixed-Precision Fine-Tuning
</h1>

<div align='center' style="font-size:18px;">
<p>
    <a href="./paper.pdf">
      <img src="https://img.shields.io/badge/Paper-ACL%202026-blue" alt="Paper"/>
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/Status-Under%20Review-orange" alt="Status"/>
    </a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
    </a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch"/>
  </p>
</div>

## 🔥 Overview

We introduce **QR-Adaptor**, a unified framework that **jointly** optimizes per-layer quantization bit-width (`q_ℓ`) and LoRA adapter rank (`r_ℓ`) under a strict memory budget.

QR-Adaptor is motivated by the **Fidelity–Plasticity Trade-off**: a layer's capacity to adapt to new tasks (*Plasticity*) is inherently constrained by the information capacity of its frozen weights (*Fidelity*). Aggressively quantizing semantically critical layers creates an information bottleneck that no amount of adapter rank can recover, while allocating high precision to robust syntactic layers wastes memory.

By aligning resource allocation with the model's intrinsic **linguistic hierarchy**, QR-Adaptor systematically liberates memory from redundancy-heavy shallow layers and reinvests it in capacity-critical deep layers. Under a **4-bit memory budget**, QR-Adaptor achieves performance rivaling 16-bit baselines, establishing a new Pareto frontier across Qwen-3 and LLaMA-3 families (1B → 14B).

### Three-Phase Framework

| Phase | Role | Module |
|-------|------|--------|
| **I. Fidelity Sensitivity Profiling** | Orthogonal sensitivity profiles `I_q(ℓ)` (Fidelity) and `I_r(ℓ)` (Plasticity); warm-start from importance priors (Eq. 6) | `qr_adaptor.core.importance` + `qr_adaptor.search.operators` |
| **II. Discrete Landscape Exploration** | Constrained NSGA-II with sensitivity-guided mutation, budget-balanced coupling, and multi-fidelity MLP surrogate (Eqs. 1–7) | `qr_adaptor.search.phase2_evolution` |
| **III. Bayesian Frontier Refinement** | Matérn-5/2 GP + Expected Improvement over atomic-edit trust regions (Eqs. 8–13) | `qr_adaptor.search.phase3_bo` |

## 🗞️ News

- **`2026-04`**: Code and paper released.

## 🛠️ Installation

```bash
conda create -n qr_adaptor python=3.10 -y
conda activate qr_adaptor

git clone https://github.com/harrysyz99/qr_adapter.git
cd qr_adapter
pip install -e .
```

This installs the `qr_adaptor` Python package and a `qr-adaptor` console entry point.

## 🚀 Quick Start

### CLI

```bash
qr-adaptor \
    --importance_json sensitivity/Qwen3-4B_dataset_wikitext2_n_sample_128_seqlen_2048.json \
    --num_layers 36 \
    --bits 2 3 4 8 \
    --ranks 4 8 16 \
    --budget_bytes 4e9 \
    --phase2_pop 40 --phase2_generations 12 --phase2_promote 6 \
    --phase3_alpha 0.6 \
    --outdir ./results/qwen3_4b
```

Equivalent helper scripts:

```bash
bash scripts/run_search.sh             # proxy mode (no GPU)
bash scripts/run_search_real_task.sh   # full SFT + lm-eval (needs GPU)
bash scripts/ablation_importance.sh    # I_q / I_r orthogonality figure
```

### Python API

```python
from qr_adaptor import QRAdaptorConfig, QRAdaptor

cfg = QRAdaptorConfig(
    num_layers=36,
    Q=[2, 3, 4, 8],
    R=[4, 8, 16],
    seed=42,
    budget_bytes=4e9,
)

runner = QRAdaptor(
    cfg,
    importance_json="sensitivity/Qwen3-4B_dataset_wikitext2_n_sample_128_seqlen_2048.json",
    target_avg_bits=4.0,
)
result = runner.run(
    outdir="./results",
    phase2_kwargs=dict(
        pop_size=40, generations=12, promote_k=6, gamma=1.5,
        lf_eval_mode="proxy",
    ),
    phase3_alpha=0.6,
)
print(result["phase3_best"])
```

### Outputs

| File | Content |
|------|---------|
| `phase2_pareto.json`  | Pareto front from Phase II evolutionary search |
| `phase3_selected.json`| Single operating point `(q*, r*)` from Phase III BO |
| `phase3_history.json` | Per-iteration BO trajectory |

## 📂 Repository Layout

```
qr_adapter/
├── qr_adaptor/                       # Main Python package
│   ├── core/                         #   Data structures (config, importance, memory, pareto)
│   ├── evaluation/                   #   Proxy + real-task evaluators
│   ├── surrogate/                    #   Multi-fidelity MLP surrogate (Eq. 7)
│   ├── search/                       #   Phase II evolution + Phase III BO + operators
│   ├── training/                     #   HQQ quantization + per-layer LoRA SFT
│   ├── experiments/                  #   lm-eval harness + ablations
│   ├── cli.py                        #   `qr-adaptor` CLI entry point
│   └── qradaptor.py                  #   Top-level Phase I / II / III orchestrator
├── scripts/                          # Reproducible bash launchers
├── examples/                         # Self-contained Python examples
├── configs/                          # YAML configs (qwen3_4b, llama3_8b)
├── sensitivity/                      # Pre-computed layer sensitivity scores
├── tests/                            # Pytest smoke tests for the public API
├── docs/                             # Architecture documentation
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

## 🧪 Testing

```bash
pip install pytest
pytest tests/ -v
```

The smoke tests cover all public dataclasses, search operators, the
Pareto/hypervolume utilities, and an end-to-end Phase II + III run on a
small synthetic problem.

## ⭐️ Citation

If you find this project useful, please cite us:

```bibtex
@inproceedings{qradaptor2026,
  title     = {Balancing Fidelity and Plasticity: Aligning Mixed-Precision
               Fine-Tuning with Linguistic Hierarchies},
  author    = {Anonymous},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for
               Computational Linguistics (ACL)},
  year      = {2026}
}
```

## 🤝 Acknowledgement

This project builds upon [HQQ](https://github.com/mobiusml/hqq), [PEFT](https://github.com/huggingface/peft), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [pymoo](https://github.com/anyoptimization/pymoo), and [scikit-learn](https://github.com/scikit-learn/scikit-learn). We thank the authors of these projects.

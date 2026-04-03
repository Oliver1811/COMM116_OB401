# Vision Agent with Dynamic Tool Generation

AI agent that answers visual questions by generating and executing Python code.  
**Model:** Qwen2-VL-2B-Instruct (4-bit, ~2.5 GB VRAM)

## Setup

**Using pip:**
```bash
git clone https://github.com/Oliver1811/COMM116.git
cd COMM116
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Using conda:**
```bash
git clone https://github.com/Oliver1811/COMM116.git
cd COMM116
conda env create -f environment.yml
conda activate vision-agent
```

**Note:** The Qwen2-VL-2B-Instruct model (~2.5 GB) downloads automatically from HuggingFace on first run. Internet connection required for initial setup only. Once downloaded, the agent operates fully offline (no online search or external APIs used during task solving).

**Verify setup:**
```bash
python verify_setup.py  # Downloads model and checks dependencies
```

## Usage

```bash
# Run evaluation (agent + baseline)
python run_eval.py --data dev.jsonl --out outputs/dev_run

# Test set
python run_eval.py --data test.jsonl --out outputs/test_run

# Quick test (3 samples)
python run_eval.py --data dev.jsonl --out outputs/test --max-samples 3
```

## Outputs

- `predictions.jsonl` — predictions per sample
- `metrics.json` — accuracy, runtime, steps
- `traces/<id>/trace.json` — step-by-step execution trace

## Key Files

- `agent2.py` — Router-based agent with task-specific code generation
- `baseline.py` — Single-pass VQA baseline
- `sandbox.py` — Restricted Python execution environment
- `model_loader.py` — Vision model loader (Qwen2-VL-2B)
- `run_eval.py` — Evaluation harness

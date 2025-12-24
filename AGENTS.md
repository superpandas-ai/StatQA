# Repository Guidelines

## Project Structure & Module Organization
- `StatQA/`: released benchmark datasets (CSV/JSON).
- `Data/`: intermediate and integrated datasets produced during construction.
- `Script/`: shell entrypoints for data extraction, evaluation, and analysis.
- `Evaluation/`: Python evaluation code for GPT and LLaMA models.
- `Finetuning/`: LLaMA-Factory configs, data, and scripts for fine-tuning.
- `Model Answer/`: generated model responses and artifacts.
- `Chart/` and `Construction/` and `Extraction/`: figures and pipeline assets.
- Top-level Python utilities: `utils.py`, `mappings.py`, `stats_dataset_features.py`.

## Build, Test, and Development Commands
- Setup environment:
  - `conda create --name statqa python=3.10.13`
  - `pip install -r requirements.txt`
- Configure API access: set your OpenAI key in `gpt_config.txt`.
- Build datasets (long-running):
  - `sh Script/info_extraction.sh`
  - `sh Script/benchmark_construction.sh`
- Run evaluations:
  - `sh Script/prompt_organization.sh`
  - `python Evaluation/gpt_evaluation.py --selected_model gpt-3.5-turbo --trick zero-shot`
  - `python Evaluation/llama_evaluation.py --model_type 2_7b --trick zero-shot`
- Analysis:
  - `sh Script/answer_analysis.sh`

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8–style naming (`snake_case` for functions/vars).
- Shell scripts: keep flags and paths explicit; prefer `Script/` as the entrypoint.
- Data paths are case- and space-sensitive (e.g., `Data/Integrated Dataset/...`).

## Testing Guidelines
- No automated test suite is defined in this repo.
- When changing evaluation logic, validate by running a small subset of data and
  confirming outputs in `Model Answer/Origin Answer`.

## Commit & Pull Request Guidelines
- Recent commits use short, imperative messages (e.g., “Update README.md”).
- Keep commits scoped to one change area; include a brief description of what
  changed and why.
- PRs should include: purpose, key files touched, and any command outputs if you
  ran long-running scripts.

## Configuration & Security Notes
- `gpt_config.txt` contains API endpoint + key; do not commit secrets.
- Model paths for LLaMA evaluation live in `Evaluation/llama_evaluation.py`.
- Fine-tuning configs are in `Finetuning/Config/`; update `model_name_or_path`
  and `output_dir` before running.

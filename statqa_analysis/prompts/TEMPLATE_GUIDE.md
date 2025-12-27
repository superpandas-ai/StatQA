# Custom Prompt Templates

## Overview

The prompt builder supports custom templates via JSON files. Templates are stored with the prompt version in `StatDatasets/prompts/<dataset-id>/<version>/template.json`.

## Template JSON Format

Create a JSON file with the following structure:

```json
{
  "task_description": "You need to select relevant columns and all applicable methods from provided list for the given statistical question.",
  "instruction": "You should only reply with one answer in JSON format containing two keys: 'columns' and 'methods'...",
  "classification": "Correlation Analysis: Pearson Correlation Coefficient, Spearman Correlation Coefficient...",
  "response": "The answer of relevant columns and applicable methods in JSON format is:",
  "cot": "Let's work this out in a step-by-step way to be sure we have the right answer. ",
  "response_explain": "You should briefly explain the reason and reply with one answer in JSON format...",
  "stats_prompt": "Methods and applicable usage scenarios:\n# Correlation Analysis\n..."
}
```

### Required Fields
- `task_description`: Task description text
- `instruction`: Instruction text
- `classification`: Classification list of methods
- `response`: Response prompt text

### Optional Fields
- `cot`: Chain-of-thought prompt (for CoT tricks)
- `response_explain`: Response prompt with explanation (for CoT tricks)
- `stats_prompt`: Enhanced stats prompt (for stats-prompt trick)

## Usage

### Using a Custom Template

```bash
# Create your template.json file
cat > my_template.json << EOF
{
  "task_description": "Custom task description...",
  "instruction": "Custom instruction...",
  "classification": "Correlation Analysis: ...",
  "response": "The answer is:"
}
EOF

# Build prompts with custom template
python -m statqa_analysis prompts build \
  --dataset-id mini-statqa \
  --prompt-version v2 \
  --trick zero-shot \
  --template my_template.json
```

The template will be copied to `StatDatasets/prompts/mini-statqa/v2/template.json`.

## Manual Trick with F-Strings

The `manual` trick allows you to build prompts using f-strings with only `meta_info_list` and `refined_question`.

### Usage

Create a text file with your f-string template:

```bash
# Create your manual template file
cat > my_manual_template.txt << EOF
Column Information:
{meta_info_list}

Question: {refined_question}

Answer in JSON format:
EOF

# Build prompts using the template file
python -m statqa_analysis prompts build \
  --dataset-id mini-statqa \
  --prompt-version v2 \
  --trick manual \
  --manual-template my_manual_template.txt
```

### Available Variables
- `{meta_info_list}`: Formatted column metadata (one per line)
- `{refined_question}`: The statistical question

### Example Manual Template File

Create `my_template.txt`:
```
You are a statistical expert. Given the following column information:

{meta_info_list}

Please answer this question: {refined_question}

Provide your answer in JSON format with 'columns' and 'methods' keys.
```

Then use it:
```bash
python -m statqa_analysis prompts build \
  --dataset-id mini-statqa \
  --prompt-version v2 \
  --trick manual \
  --manual-template my_template.txt
```

The manual template file is copied to `StatDatasets/prompts/<dataset-id>/<version>/manual_template.txt` for provenance.

## Template Storage

When using a custom template:
- Template JSON is copied to: `StatDatasets/prompts/<dataset-id>/<version>/template.json`
- Manual template (if used) is saved to: `StatDatasets/prompts/<dataset-id>/<version>/manual_template.txt`
- Manifest includes `template_source` field indicating the template path

## Default Template

If no `--template` is provided, the system uses the default template from `prompt_wording.py` constants.


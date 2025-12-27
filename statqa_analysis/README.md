# StatQA Analysis Framework - Documentation Index

## 📚 Documentation Overview

This directory contains comprehensive documentation for the StatQA Analysis Framework. Start here to find what you need.

## 🚀 Getting Started

1. **New to the framework?** → Start with [`QUICKSTART_ANALYSIS.md`](QUICKSTART_ANALYSIS.md)
2. **Want the complete workflow?** → Read [`UNIFIED_WORKFLOW.md`](UNIFIED_WORKFLOW.md) ⭐
3. **Need command reference?** → Check [`CLI_REFERENCE.md`](CLI_REFERENCE.md)

## 📖 Documentation Files

### Core Guides

- **[UNIFIED_WORKFLOW.md](UNIFIED_WORKFLOW.md)** ⭐
  - Complete end-to-end guide
  - Dataset import → Prompts → Inference → Analysis
  - Examples and best practices

- **[QUICKSTART_ANALYSIS.md](QUICKSTART_ANALYSIS.md)**
  - Quick start guide (2 minutes)
  - Basic usage examples
  - Common issues and solutions

- **[CLI_REFERENCE.md](CLI_REFERENCE.md)**
  - Complete CLI command reference
  - All options and flags
  - Workflow examples

### Technical Documentation

- **[ANALYSIS_FRAMEWORK.md](ANALYSIS_FRAMEWORK.md)**
  - Framework architecture
  - Analysis pipeline details
  - Extending the framework
  - Plugin system

- **[prompts/TEMPLATE_GUIDE.md](prompts/TEMPLATE_GUIDE.md)**
  - Custom prompt templates
  - JSON template format
  - Manual f-string templates
  - Template examples

### Implementation Details

- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)**
  - Implementation summary
  - Version history
  - Component overview

## 🎯 Quick Navigation by Task

### I want to...

**Import a dataset**
→ [`UNIFIED_WORKFLOW.md`](UNIFIED_WORKFLOW.md#step-1-import-a-dataset) or [`CLI_REFERENCE.md`](CLI_REFERENCE.md#dataset-management)

**Generate prompts**
→ [`UNIFIED_WORKFLOW.md`](UNIFIED_WORKFLOW.md#step-2-build-prompts) or [`prompts/TEMPLATE_GUIDE.md`](prompts/TEMPLATE_GUIDE.md)

**Run model inference**
→ [`UNIFIED_WORKFLOW.md`](UNIFIED_WORKFLOW.md#step-3-run-model-inference) or [`CLI_REFERENCE.md`](CLI_REFERENCE.md#model-inference)

**Analyze model outputs**
→ [`QUICKSTART_ANALYSIS.md`](QUICKSTART_ANALYSIS.md) or [`CLI_REFERENCE.md`](CLI_REFERENCE.md#analysis)

**Compare multiple models**
→ [`QUICKSTART_ANALYSIS.md`](QUICKSTART_ANALYSIS.md#-cohort-analysis-compare-multiple-models) or [`CLI_REFERENCE.md`](CLI_REFERENCE.md#cohort-analysis)

**Use custom templates**
→ [`prompts/TEMPLATE_GUIDE.md`](prompts/TEMPLATE_GUIDE.md)

**Extend the framework**
→ [`ANALYSIS_FRAMEWORK.md`](ANALYSIS_FRAMEWORK.md#extending-the-framework)

## 📂 Directory Structure

```
statqa_analysis/
├── Documentation
│   ├── UNIFIED_WORKFLOW.md      ⭐ Main guide
│   ├── QUICKSTART_ANALYSIS.md    Quick start
│   ├── CLI_REFERENCE.md          Command reference
│   ├── ANALYSIS_FRAMEWORK.md     Architecture
│   └── prompts/
│       └── TEMPLATE_GUIDE.md     Custom templates
│
├── Core Code
│   ├── cli.py                    CLI interface
│   ├── analyzer.py               Main analyzers
│   ├── config.py                 Configuration
│   ├── pipeline.py               Analysis pipeline
│   ├── datasets/                 Dataset management
│   ├── prompts/                  Prompt generation
│   ├── inference/                Model inference
│   └── analyses/                 Analysis modules
│
└── Data Directories
    ├── StatDatasets/             Dataset assets
    └── AnalysisOutput/           Run outputs
```

## 🔗 External References

- **Main README**: `../README.md` (project overview)
- **Repository Guidelines**: See workspace rules

## 💡 Tips

1. **First time?** Read `UNIFIED_WORKFLOW.md` for the complete picture
2. **Just analyzing?** Start with `QUICKSTART_ANALYSIS.md`
3. **Customizing?** Check `prompts/TEMPLATE_GUIDE.md` and `ANALYSIS_FRAMEWORK.md`
4. **Troubleshooting?** See "Common Issues" in `QUICKSTART_ANALYSIS.md`

---

**Last Updated**: December 2025  
**Framework Version**: 2.0.0


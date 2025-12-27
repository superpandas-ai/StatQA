#!/bin/bash

# ⚠️  DEPRECATED: This script uses the legacy analysis approach.
# Please use the new unified framework instead:
#
#   sh Script/model_answer_analysis.sh
#
# Or directly:
#   python -m statqa_analysis run --input <file> --out AnalysisOutput
#   python -m statqa_analysis cohort --input-dir AnalysisOutput/runs --out AnalysisOutput
#
# The legacy approach below is kept for backward compatibility.

cd ..

wait
python analyze_model_answer.py
wait
python 'Model Answer/Task Performance/summary_performance.py'
wait
python 'Model Answer/Task Performance/draw_radar_chart.py'
wait
python 'Model Answer/Task Performance/task_confusion_analysis.py'
wait
python 'Model Answer/Task Performance/error_type_analysis.py'
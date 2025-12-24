# -*- coding: utf-8 -*-
"""
Enhanced Scoring for StressQA

This module extends analyze_model_answer.py scoring capabilities to handle:
- Acceptable method sets (not just exact matching)
- Applicability checks
- Numeric tolerances for p-values, statistics, effect sizes
- Decision-quality scoring (post-hoc, corrections, confounding)
- Audit trail completeness
- Breakdown by difficulty_axes
"""

import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from test_spec_registry import get_registry


class StressQAScorer:
    """Enhanced scorer for StressQA benchmark"""
    
    def __init__(self, 
                 p_value_tolerance: float = 0.01,
                 statistic_rel_tolerance: float = 0.05,
                 effect_size_rel_tolerance: float = 0.10):
        """
        Initialize scorer with tolerance parameters.
        
        Args:
            p_value_tolerance: Absolute tolerance for p-value comparison
            statistic_rel_tolerance: Relative tolerance for test statistic
            effect_size_rel_tolerance: Relative tolerance for effect size
        """
        self.p_value_tol = p_value_tolerance
        self.stat_rel_tol = statistic_rel_tolerance
        self.es_rel_tol = effect_size_rel_tolerance
        self.registry = get_registry()
    
    def score_column_selection(self, model_cols: List[str], 
                               ground_truth_cols: List[str]) -> Dict[str, Any]:
        """
        Score column selection (exact + F1).
        
        Returns:
            Dict with 'exact', 'precision', 'recall', 'f1'
        """
        model_set = set(c.lower().strip() for c in model_cols)
        gt_set = set(c.lower().strip() for c in ground_truth_cols)
        
        if len(gt_set) == 0:
            return {'exact': 1.0 if len(model_set) == 0 else 0.0, 
                   'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        correct = len(model_set & gt_set)
        wrong = len(model_set - gt_set)
        missed = len(gt_set - model_set)
        
        exact = 1.0 if (correct == len(gt_set) and wrong == 0) else 0.0
        
        precision = correct / len(model_set) if len(model_set) > 0 else 0.0
        recall = correct / len(gt_set) if len(gt_set) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'exact': exact,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'correct': correct,
            'wrong': wrong,
            'missed': missed,
        }
    
    def score_method_selection(self, model_methods: List[str],
                               acceptable_method_sets: List[List[str]]) -> Dict[str, Any]:
        """
        Score method selection against acceptable sets.
        
        Returns:
            Dict with 'acceptable', 'exact_match', 'matched_set_index'
        """
        if not acceptable_method_sets:
            return {'acceptable': False, 'exact_match': False, 'matched_set_index': None}
        
        model_set = set(m.lower().strip() for m in model_methods)
        
        for idx, acceptable_set in enumerate(acceptable_method_sets):
            acc_set = set(m.lower().strip() for m in acceptable_set)
            
            # Check if model methods are a subset of (or equal to) acceptable set
            if model_set.issubset(acc_set) and len(model_set) > 0:
                exact = (model_set == acc_set)
                return {
                    'acceptable': True,
                    'exact_match': exact,
                    'matched_set_index': idx,
                    'model_methods': list(model_set),
                    'matched_acceptable_set': list(acc_set),
                }
        
        # No match found
        return {
            'acceptable': False,
            'exact_match': False,
            'matched_set_index': None,
            'model_methods': list(model_set),
        }
    
    def score_applicability(self, model_applicable: bool,
                           ground_truth_applicable: bool) -> Dict[str, Any]:
        """
        Score applicability judgment.
        
        Returns:
            Dict with 'correct', 'model_answer', 'ground_truth'
        """
        correct = (model_applicable == ground_truth_applicable)
        
        return {
            'correct': correct,
            'model_answer': model_applicable,
            'ground_truth': ground_truth_applicable,
        }
    
    def score_numeric_results(self, model_results: Dict[str, Any],
                             oracle_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score numeric outputs (p-value, statistic, effect size, CI) with tolerances.
        
        Returns:
            Dict with scores for each numeric field
        """
        scores = {}
        
        # P-value
        if 'p_value' in oracle_results and oracle_results['p_value'] is not None:
            model_p = model_results.get('p_value')
            oracle_p = oracle_results['p_value']
            
            if model_p is not None:
                p_close = abs(model_p - oracle_p) <= self.p_value_tol
                scores['p_value'] = {
                    'correct': p_close,
                    'model': model_p,
                    'oracle': oracle_p,
                    'diff': abs(model_p - oracle_p),
                }
            else:
                scores['p_value'] = {'correct': False, 'model': None, 'oracle': oracle_p}
        
        # Test statistic
        if 'statistic' in oracle_results and oracle_results['statistic'] is not None:
            model_stat = model_results.get('statistic')
            oracle_stat = oracle_results['statistic']
            
            if model_stat is not None and oracle_stat != 0:
                rel_diff = abs(model_stat - oracle_stat) / abs(oracle_stat)
                stat_close = rel_diff <= self.stat_rel_tol
                scores['statistic'] = {
                    'correct': stat_close,
                    'model': model_stat,
                    'oracle': oracle_stat,
                    'rel_diff': rel_diff,
                }
            elif model_stat is not None:
                scores['statistic'] = {
                    'correct': abs(model_stat - oracle_stat) < 0.01,
                    'model': model_stat,
                    'oracle': oracle_stat,
                }
            else:
                scores['statistic'] = {'correct': False, 'model': None, 'oracle': oracle_stat}
        
        # Effect size
        if 'effect_size' in oracle_results and oracle_results['effect_size'] is not None:
            model_es = model_results.get('effect_size', {})
            if isinstance(model_es, dict):
                model_es_val = model_es.get('value')
            else:
                model_es_val = model_es
            
            oracle_es = oracle_results['effect_size']
            
            if model_es_val is not None and oracle_es != 0:
                rel_diff = abs(model_es_val - oracle_es) / abs(oracle_es)
                es_close = rel_diff <= self.es_rel_tol
                scores['effect_size'] = {
                    'correct': es_close,
                    'model': model_es_val,
                    'oracle': oracle_es,
                    'rel_diff': rel_diff,
                }
            elif model_es_val is not None:
                scores['effect_size'] = {
                    'correct': abs(model_es_val - oracle_es) < 0.01,
                    'model': model_es_val,
                    'oracle': oracle_es,
                }
            else:
                scores['effect_size'] = {'correct': False, 'model': None, 'oracle': oracle_es}
        
        # Confidence interval
        if 'ci_lower' in oracle_results and oracle_results['ci_lower'] is not None:
            model_ci = model_results.get('ci', {})
            if isinstance(model_ci, dict):
                model_lower = model_ci.get('lower')
                model_upper = model_ci.get('upper')
            else:
                model_lower, model_upper = None, None
            
            oracle_lower = oracle_results['ci_lower']
            oracle_upper = oracle_results['ci_upper']
            
            ci_correct = False
            if model_lower is not None and model_upper is not None:
                # Check if CIs overlap substantially
                lower_close = abs(model_lower - oracle_lower) <= 0.1 * abs(oracle_upper - oracle_lower)
                upper_close = abs(model_upper - oracle_upper) <= 0.1 * abs(oracle_upper - oracle_lower)
                ci_correct = lower_close and upper_close
            
            scores['confidence_interval'] = {
                'correct': ci_correct,
                'model': {'lower': model_lower, 'upper': model_upper},
                'oracle': {'lower': oracle_lower, 'upper': oracle_upper},
            }
        
        return scores
    
    def score_decision_quality(self, model_answer: Dict[str, Any],
                               oracle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score decision-quality aspects:
        - Post-hoc tests recommended when needed
        - Multiple testing correction applied
        - Warnings about assumptions
        
        Returns:
            Dict with decision quality scores
        """
        scores = {}
        
        # Post-hoc
        oracle_post_hoc = oracle_data.get('post_hoc_results')
        model_post_hoc = model_answer.get('post_hoc')
        
        if oracle_post_hoc is not None:
            # Oracle says post-hoc is needed
            post_hoc_mentioned = model_post_hoc is not None and model_post_hoc != {}
            scores['post_hoc_recommended'] = {
                'correct': post_hoc_mentioned,
                'oracle_needed': True,
                'model_provided': post_hoc_mentioned,
            }
        else:
            # Post-hoc not needed
            post_hoc_not_mentioned = model_post_hoc is None or model_post_hoc == {}
            scores['post_hoc_recommended'] = {
                'correct': post_hoc_not_mentioned,
                'oracle_needed': False,
                'model_provided': not post_hoc_not_mentioned,
            }
        
        # Multiple testing correction
        oracle_correction = oracle_data.get('correction_applied')
        model_correction = model_answer.get('corrections')
        
        if oracle_correction:
            correction_applied = model_correction is not None and model_correction != ""
            scores['multiple_testing_correction'] = {
                'correct': correction_applied,
                'oracle_needed': True,
                'model_provided': correction_applied,
            }
        else:
            correction_not_applied = model_correction is None or model_correction == ""
            scores['multiple_testing_correction'] = {
                'correct': correction_not_applied,
                'oracle_needed': False,
                'model_provided': not correction_not_applied,
            }
        
        # Warnings about assumptions
        oracle_warnings = oracle_data.get('warnings', [])
        model_warnings = model_answer.get('warnings', [])
        
        if oracle_warnings:
            warnings_provided = len(model_warnings) > 0
            scores['assumption_warnings'] = {
                'correct': warnings_provided,
                'oracle_count': len(oracle_warnings),
                'model_count': len(model_warnings),
            }
        else:
            scores['assumption_warnings'] = {
                'correct': True,  # No warnings needed
                'oracle_count': 0,
                'model_count': len(model_warnings),
            }
        
        return scores
    
    def score_audit_trail(self, model_answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score completeness of audit trail.
        
        Returns:
            Dict with audit trail scores
        """
        audit_trail = model_answer.get('audit_trail', {})
        
        if not isinstance(audit_trail, dict):
            return {
                'present': False,
                'completeness': 0.0,
            }
        
        expected_fields = ['prerequisite_checks', 'method_choice_reason', 'alternatives_rejected']
        present_fields = [f for f in expected_fields if f in audit_trail and audit_trail[f]]
        
        completeness = len(present_fields) / len(expected_fields)
        
        return {
            'present': len(present_fields) > 0,
            'completeness': completeness,
            'fields_present': present_fields,
            'fields_missing': [f for f in expected_fields if f not in present_fields],
        }
    
    def score_single_answer(self, model_answer_str: str,
                           ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single model answer against ground truth.
        
        Args:
            model_answer_str: JSON string from model
            ground_truth: Dict with 'columns', 'methods', 'oracle', 'acceptable_methods', etc.
        
        Returns:
            Comprehensive score dictionary
        """
        # Parse model answer
        try:
            model_answer = json.loads(model_answer_str)
        except:
            return {
                'valid_json': False,
                'error': 'Invalid JSON',
            }
        
        scores = {'valid_json': True}
        
        # Parse ground truth components
        gt_columns = ground_truth.get('columns', [])
        gt_acceptable_methods = ground_truth.get('acceptable_methods', [])
        gt_oracle = ground_truth.get('oracle', {})
        gt_applicable = ground_truth.get('is_applicable', True)
        
        # 1. Column selection
        model_cols = model_answer.get('columns', [])
        scores['column_selection'] = self.score_column_selection(model_cols, gt_columns)
        
        # 2. Method selection
        model_methods = model_answer.get('methods', [])
        scores['method_selection'] = self.score_method_selection(model_methods, gt_acceptable_methods)
        
        # 3. Applicability
        model_applicable = model_answer.get('applicability', True)
        scores['applicability'] = self.score_applicability(model_applicable, gt_applicable)
        
        # 4. Numeric results (only if applicable)
        if gt_applicable and model_applicable:
            model_test_result = model_answer.get('test_result', {})
            scores['numeric_results'] = self.score_numeric_results(model_test_result, gt_oracle)
        else:
            scores['numeric_results'] = {}
        
        # 5. Decision quality
        scores['decision_quality'] = self.score_decision_quality(model_answer, gt_oracle)
        
        # 6. Audit trail
        scores['audit_trail'] = self.score_audit_trail(model_answer)
        
        # Compute overall score (weighted)
        weights = {
            'column_selection': 0.20,
            'method_selection': 0.25,
            'applicability': 0.10,
            'numeric_results': 0.20,
            'decision_quality': 0.15,
            'audit_trail': 0.10,
        }
        
        component_scores = {
            'column_selection': scores['column_selection']['exact'],
            'method_selection': 1.0 if scores['method_selection']['acceptable'] else 0.0,
            'applicability': 1.0 if scores['applicability']['correct'] else 0.0,
            'numeric_results': self._aggregate_numeric_score(scores['numeric_results']),
            'decision_quality': self._aggregate_decision_quality_score(scores['decision_quality']),
            'audit_trail': scores['audit_trail']['completeness'],
        }
        
        overall = sum(component_scores[k] * weights[k] for k in weights)
        
        scores['component_scores'] = component_scores
        scores['overall_score'] = overall
        
        return scores
    
    def _aggregate_numeric_score(self, numeric_scores: Dict) -> float:
        """Aggregate numeric result scores"""
        if not numeric_scores:
            return 1.0  # No numeric results expected
        
        scores = [v['correct'] for v in numeric_scores.values() if isinstance(v, dict) and 'correct' in v]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _aggregate_decision_quality_score(self, decision_scores: Dict) -> float:
        """Aggregate decision quality scores"""
        if not decision_scores:
            return 1.0
        
        scores = [v['correct'] for v in decision_scores.values() if isinstance(v, dict) and 'correct' in v]
        return sum(scores) / len(scores) if scores else 0.0
    
    def analyze_by_difficulty_axes(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate scores by difficulty_axes for breakdown analysis.
        
        Args:
            results_df: DataFrame with columns including 'difficulty_axes', 'overall_score', etc.
        
        Returns:
            DataFrame with aggregated statistics by difficulty axis
        """
        # Parse difficulty_axes from JSON strings
        def extract_axes(axes_str):
            try:
                axes = json.loads(axes_str)
                return axes if axes else ['baseline']
            except:
                return ['baseline']
        
        results_df['axes_list'] = results_df['difficulty_axes'].apply(extract_axes)
        
        # Explode so each axis gets its own row
        exploded = results_df.explode('axes_list')
        
        # Group by axis
        axis_stats = exploded.groupby('axes_list').agg({
            'overall_score': ['mean', 'std', 'count'],
        }).round(4)
        
        axis_stats.columns = ['mean_score', 'std_score', 'n_cases']
        axis_stats = axis_stats.reset_index()
        axis_stats.columns = ['difficulty_axis', 'mean_score', 'std_score', 'n_cases']
        
        return axis_stats.sort_values('mean_score')


if __name__ == "__main__":
    # Test scorer
    print("=" * 60)
    print("Testing StressQA Scorer")
    print("=" * 60)
    
    scorer = StressQAScorer()
    
    # Mock ground truth
    gt = {
        'columns': ['age', 'group'],
        'acceptable_methods': [['Independent Samples t-test', 'Welch t-test']],
        'oracle': {
            'statistic': 2.45,
            'p_value': 0.016,
            'effect_size': 0.49,
            'ci_lower': 0.12,
            'ci_upper': 1.84,
        },
        'is_applicable': True,
    }
    
    # Mock model answer
    model_json = json.dumps({
        'columns': ['age', 'group'],
        'methods': ['Welch t-test'],
        'applicability': True,
        'test_result': {'statistic': 2.47, 'p_value': 0.015},
        'audit_trail': {
            'prerequisite_checks': 'Checked normality and variance',
            'method_choice_reason': 'Selected Welch due to unequal variances',
        }
    })
    
    result = scorer.score_single_answer(model_json, gt)
    
    print(f"\n[*] Overall score: {result['overall_score']:.3f}")
    print(f"[*] Component scores:")
    for component, score in result['component_scores'].items():
        print(f"    - {component}: {score:.3f}")
    
    print("\n[✓] StressQA scorer testing complete!")


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to verify the StatQA analysis framework works.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("[*] Testing imports...")
    try:
        from statqa_analysis import ModelOutputAnalyzer, CohortAnalyzer, AnalysisConfig, AnalysisContext
        from statqa_analysis.pipeline import BaseAnalysis, AnalysisPipeline
        from statqa_analysis.io import load_csv, save_csv, safe_parse_json, safe_literal_eval
        from statqa_analysis.analyses import (
            GroundTruthDerivation,
            AnswerExtraction,
            CompareAndCount,
            Scoring,
            TaskPerformance,
            ConfusionMatrixAnalysis,
            ErrorTypeAnalysis,
        )
        print("[+] All imports successful")
        return True
    except Exception as e:
        print(f"[!] Import failed: {e}")
        return False


def test_config():
    """Test configuration objects."""
    print("\n[*] Testing configuration...")
    try:
        from statqa_analysis import AnalysisConfig, AnalysisContext
        
        config = AnalysisConfig(
            input_csv=Path("test.csv"),
            output_dir=Path("TestOutput"),
            run_id="test_run"
        )
        
        assert config.run_id == "test_run"
        assert config.method_metric == "acc"
        assert config.enable_plots == True
        
        context = AnalysisContext(config=config)
        context.add_result("test", 123)
        assert context.get_result("test") == 123
        
        print("[+] Configuration tests passed")
        return True
    except Exception as e:
        print(f"[!] Configuration test failed: {e}")
        return False


def test_pipeline():
    """Test pipeline dependency resolution."""
    print("\n[*] Testing pipeline...")
    try:
        from statqa_analysis.pipeline import BaseAnalysis, AnalysisPipeline
        from statqa_analysis import AnalysisConfig, AnalysisContext
        
        class TestAnalysis1(BaseAnalysis):
            @property
            def name(self):
                return "test1"
            
            @property
            def produces(self):
                return ["output1"]
            
            def run(self, context):
                context.add_result("output1", "value1")
                return context
        
        class TestAnalysis2(BaseAnalysis):
            @property
            def name(self):
                return "test2"
            
            @property
            def requires(self):
                return ["output1"]
            
            @property
            def produces(self):
                return ["output2"]
            
            def run(self, context):
                val1 = context.get_result("output1")
                context.add_result("output2", val1 + "_extended")
                return context
        
        config = AnalysisConfig()
        context = AnalysisContext(config=config)
        
        pipeline = AnalysisPipeline([TestAnalysis2(), TestAnalysis1()])  # Wrong order
        context = pipeline.run(context)
        
        assert context.get_result("output1") == "value1"
        assert context.get_result("output2") == "value1_extended"
        
        print("[+] Pipeline tests passed")
        return True
    except Exception as e:
        print(f"[!] Pipeline test failed: {e}")
        return False


def test_io():
    """Test I/O utilities."""
    print("\n[*] Testing I/O utilities...")
    try:
        from statqa_analysis.io import safe_parse_json, safe_literal_eval
        
        # Test JSON parsing
        result = safe_parse_json('{"key": "value"}')
        assert result == {"key": "value"}
        
        result = safe_parse_json('invalid json')
        assert result is None
        
        # Test literal eval
        result = safe_literal_eval('[1, 2, 3]')
        assert result == [1, 2, 3]
        
        result = safe_literal_eval('{"key": "value"}')
        assert result == {"key": "value"}
        
        result = safe_literal_eval('invalid')
        assert result is None
        
        print("[+] I/O tests passed")
        return True
    except Exception as e:
        print(f"[!] I/O test failed: {e}")
        return False


def test_cli_help():
    """Test that CLI help works."""
    print("\n[*] Testing CLI help...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "statqa_analysis", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0
        assert "StatQA Model Answer Analysis Framework" in result.stdout
        assert "run" in result.stdout
        assert "cohort" in result.stdout
        
        print("[+] CLI help test passed")
        return True
    except Exception as e:
        print(f"[!] CLI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("StatQA Analysis Framework - Test Suite")
    print("="*70)
    
    tests = [
        test_imports,
        test_config,
        test_pipeline,
        test_io,
        test_cli_help,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"[!] Test crashed: {e}")
            results.append(False)
    
    print("\n" + "="*70)
    print("Test Results")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


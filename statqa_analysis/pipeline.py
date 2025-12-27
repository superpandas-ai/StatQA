# -*- coding: utf-8 -*-
"""
Pipeline orchestration and base analysis interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set
from .config import AnalysisContext


class BaseAnalysis(ABC):
    """
    Base class for pluggable analyses.
    
    Each analysis declares:
    - name: unique identifier
    - requires: list of artifact/result names needed as input
    - produces: list of artifact/result names it will create
    - run(context): performs the analysis, updating context
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this analysis."""
        pass
    
    @property
    def requires(self) -> List[str]:
        """List of required inputs (artifact or result names)."""
        return []
    
    @property
    def produces(self) -> List[str]:
        """List of outputs this analysis will produce."""
        return []
    
    @abstractmethod
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """
        Execute the analysis.
        
        Args:
            context: Current analysis context with data and configuration
            
        Returns:
            Updated context with new artifacts/results
        """
        pass
    
    def can_run(self, context: AnalysisContext) -> bool:
        """Check if all required inputs are available."""
        for req in self.requires:
            if req not in context.artifacts and req not in context.results:
                return False
        return True


class AnalysisPipeline:
    """
    Orchestrates execution of analyses in dependency order.
    """
    
    def __init__(self, analyses: List[BaseAnalysis]):
        self.analyses = analyses
    
    def run(self, context: AnalysisContext, initial_resources: Optional[Set[str]] = None) -> AnalysisContext:
        """
        Run all analyses in appropriate order.
        Uses a simple topological ordering based on requires/produces.
        
        Args:
            context: Analysis context
            initial_resources: Set of resources that are already available (e.g., "raw_data")
        """
        remaining = self.analyses.copy()
        executed: Set[str] = initial_resources.copy() if initial_resources else set()
        
        while remaining:
            # Find analyses that can run now
            runnable = [a for a in remaining if self._can_execute(a, executed)]
            
            if not runnable:
                # No progress possible - check for circular dependencies or missing inputs
                missing = []
                for analysis in remaining:
                    for req in analysis.requires:
                        if req not in executed:
                            missing.append(f"{analysis.name} needs {req}")
                raise RuntimeError(
                    f"Cannot make progress in pipeline. Possible missing inputs or circular dependencies:\n" + 
                    "\n".join(missing)
                )
            
            # Execute runnable analyses
            for analysis in runnable:
                print(f"[*] Running analysis: {analysis.name}")
                try:
                    context = analysis.run(context)
                    executed.add(analysis.name)
                    for output in analysis.produces:
                        executed.add(output)
                    remaining.remove(analysis)
                    print(f"[+] Completed: {analysis.name}")
                except Exception as e:
                    print(f"[!] Error in {analysis.name}: {e}")
                    raise
        
        return context
    
    def _can_execute(self, analysis: BaseAnalysis, executed: Set[str]) -> bool:
        """Check if an analysis can execute given what's been executed."""
        for req in analysis.requires:
            if req not in executed:
                return False
        return True


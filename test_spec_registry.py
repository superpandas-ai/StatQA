# -*- coding: utf-8 -*-
"""
Test and Analysis Specification Registry for StressQA

This module defines a registry of statistical tests and analysis methods,
including their prerequisites, acceptable alternatives, and expected outputs.
This serves as the foundation for generating stress-test scenarios with
composable modifiers and layered evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum


class VariableRole(Enum):
    """Roles that variables can play in statistical analyses"""
    DV = "dependent_variable"  # Dependent/outcome variable
    IV = "independent_variable"  # Independent/predictor variable
    COVARIATE = "covariate"  # Control/adjustment variable
    STRATA = "stratification"  # Stratification variable
    CLUSTER = "cluster"  # Clustering/grouping variable
    ID = "subject_id"  # Subject/unit identifier (for repeated measures)
    TIME = "time"  # Time variable (for longitudinal)
    CONTROL = "control"  # Control variable (for partial correlation)


class VariableType(Enum):
    """Data types for variables"""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    COUNT = "count"
    BINARY = "binary"


class DesignType(Enum):
    """Study design types"""
    INDEPENDENT = "independent"
    PAIRED = "paired"
    REPEATED_MEASURES = "repeated_measures"
    CLUSTERED = "clustered"
    STRATIFIED = "stratified"


class PrerequisiteType(Enum):
    """Types of prerequisites/assumptions for statistical tests"""
    NORMALITY = "normality"
    EQUAL_VARIANCE = "equal_variance"
    MIN_SAMPLE_SIZE = "min_sample_size"
    MIN_EXPECTED_COUNT = "min_expected_count"
    INDEPENDENCE = "independence"
    NO_OUTLIERS = "no_outliers"
    LINEAR_RELATIONSHIP = "linear_relationship"
    HOMOSCEDASTICITY = "homoscedasticity"
    NO_MULTICOLLINEARITY = "no_multicollinearity"


@dataclass
class VariableSpec:
    """Specification for a variable in an analysis"""
    role: VariableRole
    var_type: VariableType
    name: Optional[str] = None
    required: bool = True


@dataclass
class Prerequisite:
    """A prerequisite/assumption for a statistical test"""
    type: PrerequisiteType
    description: str
    critical: bool = True  # If False, violation suggests robust alternative
    check_function: Optional[str] = None  # Name of function to check this


@dataclass
class OracleOutput:
    """Defines what outputs the oracle should compute"""
    statistic: bool = True
    statistic_name: Optional[str] = None  # e.g., "t", "F", "chi2"
    df: bool = False  # Degrees of freedom
    p_value: bool = True
    effect_size: bool = False
    effect_size_type: Optional[str] = None  # e.g., "cohen_d", "eta_squared", "cramers_v"
    ci: bool = False  # Confidence interval
    ci_level: float = 0.95
    post_hoc: bool = False  # Whether post-hoc tests are needed
    post_hoc_methods: List[str] = field(default_factory=list)
    multiple_testing_correction: bool = False
    additional_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSpec:
    """Complete specification for a statistical test or analysis"""
    id: str
    name: str
    family: str  # e.g., "GroupComparison", "Correlation", "DistributionTest"
    synonyms: List[str] = field(default_factory=list)
    variable_specs: List[VariableSpec] = field(default_factory=list)
    design: DesignType = DesignType.INDEPENDENT
    prerequisites: List[Prerequisite] = field(default_factory=list)
    acceptable_methods: List[List[str]] = field(default_factory=list)  # Sets of acceptable alternatives
    oracle_outputs: Optional[OracleOutput] = None
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    
    def get_primary_method(self) -> str:
        """Get the primary method name"""
        if self.acceptable_methods:
            return self.acceptable_methods[0][0] if self.acceptable_methods[0] else self.name
        return self.name
    
    def is_method_acceptable(self, method: str) -> bool:
        """Check if a method is in any of the acceptable sets"""
        method_lower = method.lower().strip()
        for method_set in self.acceptable_methods:
            if any(m.lower().strip() == method_lower for m in method_set):
                return True
        return False


class TestRegistry:
    """Registry of all test specifications"""
    
    def __init__(self):
        self._specs: Dict[str, TestSpec] = {}
        self._by_family: Dict[str, List[str]] = {}
        self._initialize_specs()
    
    def _initialize_specs(self):
        """Initialize all test specifications"""
        # Existing StatQA families
        self._add_correlation_specs()
        self._add_contingency_table_specs()
        self._add_distribution_compliance_specs()
        self._add_variance_test_specs()
        self._add_descriptive_stats_specs()
        
        # New families for stress testing
        self._add_group_comparison_specs()
        self._add_regression_ancova_specs()
        self._add_multiple_testing_specs()
    
    def register(self, spec: TestSpec):
        """Register a test specification"""
        self._specs[spec.id] = spec
        if spec.family not in self._by_family:
            self._by_family[spec.family] = []
        self._by_family[spec.family].append(spec.id)
    
    def get(self, spec_id: str) -> Optional[TestSpec]:
        """Get a test specification by ID"""
        return self._specs.get(spec_id)
    
    def get_by_family(self, family: str) -> List[TestSpec]:
        """Get all specs for a given family"""
        spec_ids = self._by_family.get(family, [])
        return [self._specs[sid] for sid in spec_ids]
    
    def all_specs(self) -> List[TestSpec]:
        """Get all registered specs"""
        return list(self._specs.values())
    
    def all_families(self) -> List[str]:
        """Get all family names"""
        return list(self._by_family.keys())
    
    # === Existing StatQA test families ===
    
    def _add_correlation_specs(self):
        """Add correlation analysis specifications"""
        # Pearson correlation
        self.register(TestSpec(
            id="pearson_correlation",
            name="Pearson Correlation Coefficient",
            family="Correlation Analysis",
            synonyms=["Pearson's r", "Pearson correlation"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.NORMALITY, "Both variables should be approximately normally distributed", critical=False),
                Prerequisite(PrerequisiteType.LINEAR_RELATIONSHIP, "Relationship should be linear", critical=True),
            ],
            acceptable_methods=[
                ["Pearson Correlation Coefficient"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="r",
                p_value=True,
                effect_size=True, effect_size_type="r",
                ci=True,
            ),
            description="Measures linear correlation between two continuous variables",
            tags={"parametric", "bivariate"}
        ))
        
        # Spearman correlation
        self.register(TestSpec(
            id="spearman_correlation",
            name="Spearman Correlation Coefficient",
            family="Correlation Analysis",
            synonyms=["Spearman's rho", "Spearman rank correlation"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Spearman Correlation Coefficient"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="rho",
                p_value=True,
                effect_size=True, effect_size_type="rho",
            ),
            description="Non-parametric rank correlation for monotonic relationships",
            tags={"nonparametric", "bivariate", "robust"}
        ))
        
        # Kendall correlation
        self.register(TestSpec(
            id="kendall_correlation",
            name="Kendall Correlation Coefficient",
            family="Correlation Analysis",
            synonyms=["Kendall's tau", "Kendall rank correlation"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Kendall Correlation Coefficient"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="tau",
                p_value=True,
            ),
            description="Non-parametric rank correlation, more robust to outliers than Spearman",
            tags={"nonparametric", "bivariate", "robust"}
        ))
        
        # Partial correlation
        self.register(TestSpec(
            id="partial_correlation",
            name="Partial Correlation Coefficient",
            family="Correlation Analysis",
            synonyms=["Partial correlation"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.CONTROL, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.NORMALITY, "Variables should be approximately normally distributed", critical=False),
                Prerequisite(PrerequisiteType.LINEAR_RELATIONSHIP, "Relationships should be linear", critical=True),
            ],
            acceptable_methods=[
                ["Partial Correlation Coefficient"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="r_partial",
                p_value=True,
                effect_size=True, effect_size_type="r_partial",
            ),
            description="Correlation between two variables controlling for a third",
            tags={"parametric", "multivariate"}
        ))
    
    def _add_contingency_table_specs(self):
        """Add contingency table test specifications"""
        # Chi-square test
        self.register(TestSpec(
            id="chi_square_independence",
            name="Chi-square Independence Test",
            family="Contingency Table Test",
            synonyms=["Chi-square test", "Chi-squared test", "χ² test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CATEGORICAL),
                VariableSpec(VariableRole.IV, VariableType.CATEGORICAL),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.MIN_EXPECTED_COUNT, "Expected count ≥ 5 in all cells", critical=True),
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
            ],
            acceptable_methods=[
                ["Chi-square Independence Test", "Chi-squared test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="chi2",
                df=True,
                p_value=True,
                effect_size=True, effect_size_type="cramers_v",
            ),
            description="Tests independence of two categorical variables",
            tags={"categorical", "independence"}
        ))
        
        # Fisher's exact test
        self.register(TestSpec(
            id="fisher_exact",
            name="Fisher Exact Test",
            family="Contingency Table Test",
            synonyms=["Fisher's exact test", "Fisher test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CATEGORICAL),
                VariableSpec(VariableRole.IV, VariableType.CATEGORICAL),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
            ],
            acceptable_methods=[
                ["Fisher Exact Test", "Fisher's exact test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=False,
                p_value=True,
                effect_size=True, effect_size_type="odds_ratio",
            ),
            description="Exact test for independence in 2×2 tables, especially with small samples",
            tags={"categorical", "exact", "small_sample"}
        ))
        
        # Mantel-Haenszel test
        self.register(TestSpec(
            id="mantel_haenszel",
            name="Mantel-Haenszel Test",
            family="Contingency Table Test",
            synonyms=["Mantel-Haenszel test", "Cochran-Mantel-Haenszel test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CATEGORICAL),
                VariableSpec(VariableRole.IV, VariableType.CATEGORICAL),
                VariableSpec(VariableRole.STRATA, VariableType.CATEGORICAL),
            ],
            design=DesignType.STRATIFIED,
            prerequisites=[
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent within strata", critical=True),
            ],
            acceptable_methods=[
                ["Mantel-Haenszel Test", "Cochran-Mantel-Haenszel test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="chi2_mh",
                df=True,
                p_value=True,
            ),
            description="Tests conditional independence across strata",
            tags={"categorical", "stratified"}
        ))
    
    def _add_distribution_compliance_specs(self):
        """Add distribution compliance test specifications"""
        # Shapiro-Wilk
        self.register(TestSpec(
            id="shapiro_wilk",
            name="Shapiro-Wilk Test of Normality",
            family="Distribution Compliance Test",
            synonyms=["Shapiro-Wilk test", "Shapiro test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Shapiro-Wilk Test of Normality", "Shapiro-Wilk test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="W",
                p_value=True,
            ),
            description="Tests if data follows a normal distribution",
            tags={"normality", "univariate"}
        ))
        
        # Anderson-Darling
        self.register(TestSpec(
            id="anderson_darling",
            name="Anderson-Darling Test",
            family="Distribution Compliance Test",
            synonyms=["Anderson-Darling test", "AD test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Anderson-Darling Test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="A2",
                p_value=False,  # Returns critical values instead
                additional_fields={"critical_values": list, "significance_levels": list}
            ),
            description="Tests goodness of fit to a distribution, more sensitive to tails than KS",
            tags={"normality", "univariate", "goodness_of_fit"}
        ))
        
        # Kolmogorov-Smirnov for normality
        self.register(TestSpec(
            id="ks_normality",
            name="Kolmogorov-Smirnov Test for Normality",
            family="Distribution Compliance Test",
            synonyms=["KS test for normality", "Kolmogorov-Smirnov normality test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Kolmogorov-Smirnov Test for Normality"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="D",
                p_value=True,
            ),
            description="Tests if data follows a normal distribution",
            tags={"normality", "univariate"}
        ))
        
        # KS test for distribution comparison
        self.register(TestSpec(
            id="ks_two_sample",
            name="Kolmogorov-Smirnov Test",
            family="Distribution Compliance Test",
            synonyms=["Two-sample KS test", "Kolmogorov-Smirnov two-sample test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS, name="sample1"),
                VariableSpec(VariableRole.IV, VariableType.CONTINUOUS, name="sample2"),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Kolmogorov-Smirnov Test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="D",
                p_value=True,
            ),
            description="Tests if two samples come from the same distribution",
            tags={"distribution_comparison", "nonparametric"}
        ))
        
        # Lilliefors test
        self.register(TestSpec(
            id="lilliefors",
            name="Lilliefors Test",
            family="Distribution Compliance Test",
            synonyms=["Lilliefors test for normality"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Lilliefors Test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="D",
                p_value=True,
            ),
            description="Variant of KS test for normality with estimated parameters",
            tags={"normality", "univariate"}
        ))
    
    def _add_variance_test_specs(self):
        """Add variance test specifications"""
        # Levene's test
        self.register(TestSpec(
            id="levene_test",
            name="Levene Test",
            family="Variance Test",
            synonyms=["Levene's test", "Levene test for equality of variances"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CATEGORICAL),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Levene Test", "Levene's test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="W",
                df=True,
                p_value=True,
            ),
            description="Tests equality of variances across groups, robust to non-normality",
            tags={"variance", "robust"}
        ))
        
        # Bartlett's test
        self.register(TestSpec(
            id="bartlett_test",
            name="Bartlett Test",
            family="Variance Test",
            synonyms=["Bartlett's test", "Bartlett test for homogeneity of variances"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CATEGORICAL),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.NORMALITY, "Data should be normally distributed in each group", critical=True),
            ],
            acceptable_methods=[
                ["Bartlett Test", "Bartlett's test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="chi2",
                df=True,
                p_value=True,
            ),
            description="Tests equality of variances, sensitive to normality assumption",
            tags={"variance", "parametric"}
        ))
        
        # F-test for variance
        self.register(TestSpec(
            id="f_test_variance",
            name="F-Test for Variance",
            family="Variance Test",
            synonyms=["F-test for equality of variances", "Variance ratio test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.BINARY),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.NORMALITY, "Data should be normally distributed in each group", critical=True),
            ],
            acceptable_methods=[
                ["F-Test for Variance"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="F",
                df=True,
                p_value=True,
            ),
            description="Tests equality of variances for two groups",
            tags={"variance", "parametric", "two_sample"}
        ))
        
        # Mood's test
        self.register(TestSpec(
            id="mood_test",
            name="Mood Variance Test",
            family="Variance Test",
            synonyms=["Mood's test", "Mood test for scale"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.BINARY),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Mood Variance Test", "Mood's test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="z",
                p_value=True,
            ),
            description="Non-parametric test for equality of scale parameters",
            tags={"variance", "nonparametric", "two_sample"}
        ))
    
    def _add_descriptive_stats_specs(self):
        """Add descriptive statistics specifications"""
        descriptive_stats = [
            ("mean", "Mean", "Arithmetic mean"),
            ("median", "Median", "Middle value"),
            ("mode", "Mode", "Most frequent value"),
            ("range", "Range", "Difference between max and min"),
            ("quartile", "Quartile", "25th, 50th, 75th percentiles"),
            ("std_dev", "Standard Deviation", "Measure of dispersion"),
            ("skewness", "Skewness", "Measure of asymmetry"),
            ("kurtosis", "Kurtosis", "Measure of tailedness"),
        ]
        
        for stat_id, stat_name, description in descriptive_stats:
            self.register(TestSpec(
                id=stat_id,
                name=stat_name,
                family="Descriptive Statistics",
                synonyms=[],
                variable_specs=[
                    VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                ],
                design=DesignType.INDEPENDENT,
                prerequisites=[],
                acceptable_methods=[[stat_name]],
                oracle_outputs=OracleOutput(
                    statistic=True, statistic_name=stat_id,
                    p_value=False,
                ),
                description=description,
                tags={"descriptive", "univariate"}
            ))
    
    # === New test families for stress testing ===
    
    def _add_group_comparison_specs(self):
        """Add group comparison test specifications"""
        # Independent t-test
        self.register(TestSpec(
            id="independent_t_test",
            name="Independent Samples t-test",
            family="Group Comparison",
            synonyms=["Two-sample t-test", "Independent t-test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.BINARY),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.NORMALITY, "Data should be approximately normal in each group", critical=False),
                Prerequisite(PrerequisiteType.EQUAL_VARIANCE, "Variances should be equal across groups", critical=False),
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
            ],
            acceptable_methods=[
                ["Independent Samples t-test", "Two-sample t-test", "Student's t-test"],
                ["Welch t-test", "Welch's t-test"],  # Alternative when equal variance violated
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="t",
                df=True,
                p_value=True,
                effect_size=True, effect_size_type="cohen_d",
                ci=True,
            ),
            description="Compares means of two independent groups",
            tags={"parametric", "two_sample", "mean_comparison"}
        ))
        
        # Welch's t-test
        self.register(TestSpec(
            id="welch_t_test",
            name="Welch t-test",
            family="Group Comparison",
            synonyms=["Welch's t-test", "Unequal variances t-test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.BINARY),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.NORMALITY, "Data should be approximately normal in each group", critical=False),
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
            ],
            acceptable_methods=[
                ["Welch t-test", "Welch's t-test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="t",
                df=True,
                p_value=True,
                effect_size=True, effect_size_type="cohen_d",
                ci=True,
            ),
            description="Compares means of two independent groups without assuming equal variances",
            tags={"parametric", "two_sample", "robust", "mean_comparison"}
        ))
        
        # Mann-Whitney U test
        self.register(TestSpec(
            id="mann_whitney",
            name="Mann-Whitney U Test",
            family="Group Comparison",
            synonyms=["Mann-Whitney test", "Wilcoxon rank-sum test", "Mann-Whitney-Wilcoxon test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.BINARY),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
            ],
            acceptable_methods=[
                ["Mann-Whitney U Test", "Mann-Whitney test", "Wilcoxon rank-sum test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="U",
                p_value=True,
                effect_size=True, effect_size_type="rank_biserial",
            ),
            description="Non-parametric test comparing distributions of two independent groups",
            tags={"nonparametric", "two_sample", "robust"}
        ))
        
        # Paired t-test
        self.register(TestSpec(
            id="paired_t_test",
            name="Paired Samples t-test",
            family="Group Comparison",
            synonyms=["Paired t-test", "Dependent t-test", "Matched pairs t-test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.BINARY),
                VariableSpec(VariableRole.ID, VariableType.CATEGORICAL),
            ],
            design=DesignType.PAIRED,
            prerequisites=[
                Prerequisite(PrerequisiteType.NORMALITY, "Differences should be approximately normal", critical=False),
            ],
            acceptable_methods=[
                ["Paired Samples t-test", "Paired t-test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="t",
                df=True,
                p_value=True,
                effect_size=True, effect_size_type="cohen_d",
                ci=True,
            ),
            description="Compares means of two related groups",
            tags={"parametric", "paired", "mean_comparison"}
        ))
        
        # Wilcoxon signed-rank test
        self.register(TestSpec(
            id="wilcoxon_signed_rank",
            name="Wilcoxon Signed-Rank Test",
            family="Group Comparison",
            synonyms=["Wilcoxon test", "Wilcoxon matched-pairs test"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.BINARY),
                VariableSpec(VariableRole.ID, VariableType.CATEGORICAL),
            ],
            design=DesignType.PAIRED,
            prerequisites=[],
            acceptable_methods=[
                ["Wilcoxon Signed-Rank Test", "Wilcoxon test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="W",
                p_value=True,
                effect_size=True, effect_size_type="rank_biserial",
            ),
            description="Non-parametric test for paired samples",
            tags={"nonparametric", "paired", "robust"}
        ))
        
        # One-way ANOVA
        self.register(TestSpec(
            id="one_way_anova",
            name="One-Way ANOVA",
            family="Group Comparison",
            synonyms=["ANOVA", "One-way analysis of variance"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CATEGORICAL),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.NORMALITY, "Data should be approximately normal in each group", critical=False),
                Prerequisite(PrerequisiteType.EQUAL_VARIANCE, "Variances should be equal across groups", critical=False),
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
            ],
            acceptable_methods=[
                ["One-Way ANOVA", "ANOVA"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="F",
                df=True,
                p_value=True,
                effect_size=True, effect_size_type="eta_squared",
                post_hoc=True,
                post_hoc_methods=["Tukey HSD", "Bonferroni", "Scheffe"],
            ),
            description="Compares means across three or more independent groups",
            tags={"parametric", "multi_group", "mean_comparison"}
        ))
        
        # Kruskal-Wallis H test
        self.register(TestSpec(
            id="kruskal_wallis",
            name="Kruskal-Wallis H Test",
            family="Group Comparison",
            synonyms=["Kruskal-Wallis test", "Kruskal-Wallis one-way ANOVA"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CATEGORICAL),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
            ],
            acceptable_methods=[
                ["Kruskal-Wallis H Test", "Kruskal-Wallis test"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="H",
                df=True,
                p_value=True,
                effect_size=True, effect_size_type="eta_squared",
                post_hoc=True,
                post_hoc_methods=["Dunn's test"],
            ),
            description="Non-parametric test comparing distributions across three or more groups",
            tags={"nonparametric", "multi_group", "robust"}
        ))
    
    def _add_regression_ancova_specs(self):
        """Add regression and ANCOVA specifications"""
        # Simple linear regression
        self.register(TestSpec(
            id="simple_linear_regression",
            name="Simple Linear Regression",
            family="Regression",
            synonyms=["Linear regression", "OLS regression"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.LINEAR_RELATIONSHIP, "Relationship should be linear", critical=True),
                Prerequisite(PrerequisiteType.NORMALITY, "Residuals should be normally distributed", critical=False),
                Prerequisite(PrerequisiteType.HOMOSCEDASTICITY, "Residuals should have constant variance", critical=False),
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
            ],
            acceptable_methods=[
                ["Simple Linear Regression", "Linear regression"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="F",
                df=True,
                p_value=True,
                effect_size=True, effect_size_type="r_squared",
                ci=True,
                additional_fields={"coefficients": dict, "se": dict, "t_values": dict, "p_values": dict}
            ),
            description="Models linear relationship between continuous predictor and outcome",
            tags={"regression", "parametric", "continuous"}
        ))
        
        # Multiple linear regression
        self.register(TestSpec(
            id="multiple_linear_regression",
            name="Multiple Linear Regression",
            family="Regression",
            synonyms=["Multiple regression", "MLR"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.COVARIATE, VariableType.CONTINUOUS, required=False),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.LINEAR_RELATIONSHIP, "Relationships should be linear", critical=True),
                Prerequisite(PrerequisiteType.NORMALITY, "Residuals should be normally distributed", critical=False),
                Prerequisite(PrerequisiteType.HOMOSCEDASTICITY, "Residuals should have constant variance", critical=False),
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
                Prerequisite(PrerequisiteType.NO_MULTICOLLINEARITY, "Predictors should not be highly correlated", critical=False),
            ],
            acceptable_methods=[
                ["Multiple Linear Regression", "Multiple regression"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="F",
                df=True,
                p_value=True,
                effect_size=True, effect_size_type="r_squared",
                ci=True,
                additional_fields={"coefficients": dict, "se": dict, "t_values": dict, "p_values": dict, "vif": dict}
            ),
            description="Models linear relationship with multiple predictors",
            tags={"regression", "parametric", "multivariate"}
        ))
        
        # ANCOVA
        self.register(TestSpec(
            id="ancova",
            name="Analysis of Covariance (ANCOVA)",
            family="Regression",
            synonyms=["ANCOVA"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.CATEGORICAL),
                VariableSpec(VariableRole.COVARIATE, VariableType.CONTINUOUS),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[
                Prerequisite(PrerequisiteType.LINEAR_RELATIONSHIP, "Covariate should be linearly related to DV", critical=True),
                Prerequisite(PrerequisiteType.NORMALITY, "Residuals should be normally distributed", critical=False),
                Prerequisite(PrerequisiteType.EQUAL_VARIANCE, "Variances should be equal across groups", critical=False),
                Prerequisite(PrerequisiteType.INDEPENDENCE, "Observations must be independent", critical=True),
            ],
            acceptable_methods=[
                ["Analysis of Covariance", "ANCOVA"],
            ],
            oracle_outputs=OracleOutput(
                statistic=True, statistic_name="F",
                df=True,
                p_value=True,
                effect_size=True, effect_size_type="partial_eta_squared",
                ci=True,
                post_hoc=True,
                post_hoc_methods=["Adjusted means comparison"],
            ),
            description="Compares group means while controlling for covariates",
            tags={"regression", "parametric", "covariate_adjustment"}
        ))
    
    def _add_multiple_testing_specs(self):
        """Add multiple testing correction specifications"""
        # Multiple endpoints comparison
        self.register(TestSpec(
            id="multiple_endpoints",
            name="Multiple Endpoints Comparison",
            family="Multiple Testing",
            synonyms=["Multiple comparisons", "Multiple outcomes"],
            variable_specs=[
                VariableSpec(VariableRole.DV, VariableType.CONTINUOUS),
                VariableSpec(VariableRole.IV, VariableType.BINARY),
            ],
            design=DesignType.INDEPENDENT,
            prerequisites=[],
            acceptable_methods=[
                ["Multiple Endpoints with Benjamini-Hochberg", "Multiple Endpoints with FDR"],
                ["Multiple Endpoints with Bonferroni"],
                ["Multiple Endpoints with Holm-Bonferroni"],
            ],
            oracle_outputs=OracleOutput(
                statistic=False,
                p_value=True,
                multiple_testing_correction=True,
                additional_fields={"raw_p_values": list, "adjusted_p_values": list, "correction_method": str}
            ),
            description="Tests multiple outcomes with appropriate correction for multiple comparisons",
            tags={"multiple_testing", "correction"}
        ))


# Global registry instance
REGISTRY = TestRegistry()


def get_registry() -> TestRegistry:
    """Get the global test registry"""
    return REGISTRY


'''
Wordings for prompt organization
'''




'''-------------------------------------------
Basic prompt for introduction and instruction
'''

PROMPT_TASK_DESCRIPTION = '''You need to select relevant columns and all applicable methods from provided list for the given statistical question.'''
PROMPT_TASK_DESCRIPTION_EXPLAIN = '''You need to select relevant columns and all applicable methods from provided list for the given statistical question, and give your reason.'''

PROMPT_INSTRUCTION = '''You should only reply with one answer in JSON format containing two keys: 'columns' and 'methods'. The value of 'columns' is a list of columns' headers relevant to the given statistical question, and the value of 'methods' is a list containing all methods you think applicable. For example: {"columns": ["c1", "c2", "..."], "methods": ["m1", "m2", "..."]}. If you think there is a strata or control variable involved, put its column header in the last item of the columns list. Ensure your methods selection is only limited to the classification list provided.'''
PROMPT_INSTRUCTION_EXPLAIN = '''You should briefly explain the reason and reply with one answer in JSON format containing two keys: 'columns' and 'methods'. The value of 'columns' is a list of columns' headers relevant to the given statistical question, and the value of 'methods' is a list containing all methods you think applicable. For example: {"columns": ["c1", "c2", "..."], "methods": ["m1", "m2", "..."]}. If you think there is a strata or control variable involved, put its column header in the last item of the columns list. Ensure your methods selection is only limited to the classification list provided.'''

PROMPT_CLASSIFICATION = '''Correlation Analysis: Pearson Correlation Coefficient, Spearman Correlation Coefficient, Kendall Correlation Coefficient, Partial Correlation Coefficient;
Distribution Compliance Test: Anderson-Darling Test, Shapiro-Wilk Test of Normality, Kolmogorov-Smirnov Test for Normality, Lilliefors Test, Kolmogorov-Smirnov Test, Kolmogorov-Smirnov Test for Uniform distribution, Kolmogorov-Smirnov Test for Gamma distribution, Kolmogorov-Smirnov Test for Exponential distribution;
Contingency Table Test: Chi-square Independence Test, Fisher Exact Test, Mantel-Haenszel Test;
Descriptive Statistics: Mean, Median, Mode, Range, Quartile, Standard Deviation, Skewness, Kurtosis;
Variance Test: Mood Variance Test, Levene Test, Bartlett Test, F-Test for Variance;
Group Comparison: Independent Samples t-test, Welch t-test, Mann-Whitney U Test, Paired Samples t-test, Wilcoxon Signed-Rank Test, One-Way ANOVA, Kruskal-Wallis H Test;
Regression: Simple Linear Regression, Multiple Linear Regression, Analysis of Covariance (ANCOVA);
Multiple Testing: Multiple Endpoints with Benjamini-Hochberg, Multiple Endpoints with Bonferroni, Multiple Endpoints with Holm-Bonferroni.'''

PROMPT_RESPONSE = '''The answer of relevant columns and applicable methods in JSON format is:'''
PROMPT_RESPONSE_EXPLAIN = '''The answer of relevant columns and applicable methods in JSON format and reason is:'''

PROMPT_COT = '''Let's work this out in a step-by-step way to be sure we have the right answer. '''


'''-------------------------------------------
Enhanced prompt for StressQA with superstat-compatible JSON contract
'''

STRESSQA_TASK_DESCRIPTION = '''You need to perform a complete statistical analysis for the given question, including selecting relevant columns, identifying all applicable methods, checking prerequisites, computing results, and providing a deterministic audit trail of your reasoning.'''

STRESSQA_INSTRUCTION = '''You should reply with a single JSON object containing the following keys:

1. "columns": List of column headers relevant to the question.
2. "methods": List of all applicable statistical methods you selected.
3. "applicability": Boolean indicating whether the analysis is applicable given the data and question.
4. "checks": Object describing assumption checks you performed (e.g., {"normality": true, "equal_variance": false}).
5. "warnings": List of warnings about violated assumptions or limitations.
6. "test_result": Object with keys "statistic", "df", "p_value" if applicable.
7. "effect_size": Object with keys "value" and "type" (e.g., cohen_d, eta_squared).
8. "ci": Object with keys "lower", "upper", "level" for confidence intervals.
9. "post_hoc": Object describing post-hoc tests if needed (e.g., {"recommended": ["Tukey HSD"], "reason": "Omnibus test significant"}).
10. "corrections": String describing multiple testing correction applied if any.
11. "audit_trail": Object with keys "prerequisite_checks", "method_choice_reason", "alternatives_rejected" providing short deterministic explanations.

Example format: 
{
  "columns": ["col1", "col2"],
  "methods": ["Independent Samples t-test"],
  "applicability": true,
  "checks": {"normality": true, "equal_variance": false},
  "warnings": ["Equal variance assumption violated; Welch t-test recommended"],
  "test_result": {"statistic": 2.45, "df": 98, "p_value": 0.016},
  "effect_size": {"value": 0.49, "type": "cohen_d"},
  "ci": {"lower": 0.12, "upper": 1.84, "level": 0.95},
  "post_hoc": null,
  "corrections": null,
  "audit_trail": {
    "prerequisite_checks": "Checked normality (Shapiro-Wilk: group1 p=0.23, group2 p=0.18) and homogeneity of variance (Levene: p=0.003)",
    "method_choice_reason": "Selected Welch t-test due to unequal variances",
    "alternatives_rejected": "Student's t-test rejected due to heteroscedasticity"
  }
}

Ensure your methods selection is limited to the classification list provided.'''

STRESSQA_RESPONSE = '''The complete statistical analysis in JSON format is:'''




'''------------------
Examples for few-shot
'''

CA_EG = '''# Column Information: 
column_header: TV Ad Budget ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
column_header: Radio Ad Budget ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
column_header: Newspaper Ad Budget ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
column_header: Sales ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
# Statistical Question: Is there a linear correlation between the TV advertising budget ($) and sales revenue ($) in this study?
# Correct Answer: {"columns": ["TV Ad Budget ($)",  "Sales ($)"], "methods": ["Pearson Correlation Coefficient", "Spearman Correlation Coefficient", "Kendall Correlation Coefficient"]}'''

CTT_EG = '''# Column Information: 
column_header: RowNumber; data_type: id; num_of_rows: 10002; is_normality: False.
column_header: CustomerId; data_type: id; num_of_rows: 10002; is_normality: False.
column_header: Surname; data_type: other; num_of_rows: 10002; is_normality: False.
column_header: CreditScore; data_type: quantitative; num_of_rows: 10002; is_normality: False.
column_header: Geography; data_type: categorical; num_of_rows: 10002; is_normality: False.
column_header: Gender; data_type: categorical; num_of_rows: 10002; is_normality: False.
column_header: Age; data_type: quantitative; num_of_rows: 10002; is_normality: False.
column_header: Tenure; data_type: quantitative; num_of_rows: 10002; is_normality: False.
column_header: Balance; data_type: quantitative; num_of_rows: 10002; is_normality: False.
column_header: NumOfProducts; data_type: quantitative; num_of_rows: 10002; is_normality: False.
column_header: HasCrCard; data_type: categorical; num_of_rows: 10002; is_normality: False.
column_header: IsActiveMember; data_type: categorical; num_of_rows: 10002; is_normality: False.
column_header: EstimatedSalary; data_type: quantitative; num_of_rows: 10002; is_normality: False.
column_header: Exited; data_type: categorical; num_of_rows: 10002; is_normality: False.
# Statistical Question: Is there a statistically significant correlation between gender and possession of a credit card among bank customers?
# Correct Answer: {"columns": ["Gender", "HasCrCard"], "methods": ["Chi-square Independence Test", "Fisher Exact Test"]}'''

DS_EG = '''
# Column Information: 
column_header: Type of tree; data_type: categorical; num_of_rows: 14; is_normality: False.
column_header: Number of trees sold; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Average Tree Price; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Sales; data_type: quantitative; num_of_rows: 14; is_normality: True.
# Statistical Question: Could you determine what is the median value of the sold trees?
# Correct Answer: {"columns": ["Number of trees sold"], "methods": ["Median"]}
'''

DCT_EG = '''# Column Information: 
column_header: index; data_type: id; num_of_rows: 14; is_normality: False.
column_header: Year; data_type: other; num_of_rows: 14; is_normality: False.
column_header: Type of tree; data_type: categorical; num_of_rows: 14; is_normality: False.
column_header: Number of trees sold; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Average Tree Price; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Sales; data_type: quantitative; num_of_rows: 14; is_normality: True.
# Statistical Question: Does the distribution of the number of trees sold data align with a Gamma distribution?
# Correct Answer: {"columns": ["Number of trees sold"], "methods": ["Kolmogorov-Smirnov Test for Gamma distribution"]}'''

VT_EG = '''# Column Information: 
column_header: index; data_type: id; num_of_rows: 14; is_normality: False.
column_header: Year; data_type: other; num_of_rows: 14; is_normality: False.
column_header: Type of tree; data_type: categorical; num_of_rows: 14; is_normality: False.
column_header: Number of trees sold; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Average Tree Price; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Sales; data_type: quantitative; num_of_rows: 14; is_normality: True.
# Statistical Question: Is the variability in the number of trees sold comparable to that of sales?
# Correct Answer: {"columns": ["Number of trees sold", "Sales"], "methods": ["Mood Variance Test", "Levene Test", "Bartlett Test", "F-Test for Variance"]}'''




'''--------------
Examples for CoT
'''

COT_CA_EG = '''# Column Information: 
column_header: TV Ad Budget ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
column_header: Radio Ad Budget ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
column_header: Newspaper Ad Budget ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
column_header: Sales ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
# Statistical Question: Is there a correlation between the TV advertising budget and sales revenue in this study?
# Correct Answer: Firstly, the question asks about TV advertising budget and sales revenue, so the relevant columns are TV Ad Budget ($) and Sales ($). Secondly, given that it asks about correlation, and the variables involved are all quantitative, some methods from Correlation Analysis can be applicable. Thirdly, there are enough samples of 200, with only two variables involved, and no control variables, so the applicable methods are Pearson Correlation Coefficient, Spearman Correlation Coefficient, Kendall Correlation Coefficient. Hence, the answer is: {"columns": ["TV Ad Budget ($)",  "Sales ($)"], "methods": ["Pearson Correlation Coefficient", "Spearman Correlation Coefficient", "Kendall Correlation Coefficient"]}'''


COT_CTT_EG = '''# Column Information: 
column_header: CustomerId; data_type: id; num_of_rows: 10002; is_normality: False.
column_header: Gender; data_type: categorical; num_of_rows: 10002; is_normality: False.
column_header: Age; data_type: quantitative; num_of_rows: 10002; is_normality: False.
column_header: Tenure; data_type: quantitative; num_of_rows: 10002; is_normality: False.
column_header: HasCrCard; data_type: categorical; num_of_rows: 10002; is_normality: False.
column_header: IsActiveMember; data_type: categorical; num_of_rows: 10002; is_normality: False.
column_header: EstimatedSalary; data_type: quantitative; num_of_rows: 10002; is_normality: False.
column_header: Exited; data_type: categorical; num_of_rows: 10002; is_normality: False.
# Statistical Question: Is there a statistically significant correlation between gender and possession of a credit card among bank customers?
# Correct Answer: Firstly, the question asks about correlation between gender and possession of a credit card, so the relevant columns are "Gender" and "HasCrCard". Secondly, data of both relevant columns is categorical, so some methods of Contingency Table Test can be applicable. Thirdly, the number of samples is as large as more than 10000, and there are only two variables involved, no strata variable, so the applicable statistical methods from the list are Chi-square Independence Test and Fisher Exact Test. Hence, the answer is: {"columns": ["Gender", "HasCrCard"], "methods": ["Chi-square Independence Test", "Fisher Exact Test"]}'''


COT_DCT_EG = '''# Column Information: 
column_header: Year; data_type: Other; num_of_rows: 4; is_normality: False.
column_header: Nigeria; data_type: quantitative; num_of_rows: 4; is_normality: True.
column_header: Pakistan; data_type: quantitative; num_of_rows: 4; is_normality: True.
column_header: Japan; data_type: quantitative; num_of_rows: 4; is_normality: True.
column_header: United States; data_type: quantitative; num_of_rows: 4; is_normality: True.
column_header: Brazil; data_type: quantitative; num_of_rows: 4; is_normality: True.
# Statistical Question: Is the annual crop production in Japan normally distributed?
# Correct Answer: Firstly, the question asks whether the distribution of grain production in Japan conforms to normality, so the relevant column is "Japan", and the type of task is the normality test in the distribution compliance test. Secondly, in terms of methods selection, column information shows that the "Japan" column is quantitative and the sample size is very small with only 4 rows (less than 50 samples). Therefore Lilliefors Test which applies to large samples should be excluded and other methods of normality test can be used, including Anderson-Darling Test, Shapiro-Wilk Test of Normality, and Kolmogorov-Smirnov Test for Normality. Hence, the answer is: {"columns": ["Japan"], "methods": ["Anderson-Darling Test", "Shapiro-Wilk Test of Normality", "Kolmogorov-Smirnov Test for Normality"]}'''


COT_VT_EG = '''# Column Information: 
column_header: index; data_type: id; num_of_rows: 14; is_normality: False.
column_header: Year; data_type: other; num_of_rows: 14; is_normality: False.
column_header: Type of tree; data_type: categorical; num_of_rows: 14; is_normality: False.
column_header: Number of trees sold; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Average Tree Price; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Sales; data_type: quantitative; num_of_rows: 14; is_normality: True.
# Statistical Question: Is the variability in how many trees are sold comparable to that of sales?
# Correct Answer: Firstly, the question asks about number of sold trees and their sales, so there are two relevant columns: Number of trees sold and Sales. Secondly, to compare two variables' variability, we should apply methods of variance test. Thirdly, data of both relevant columns are quantitative and normally distributed, so Mood Variance Test, Levene Test, Bartlett Test, and F-Test for Variance are applicable. Hence, the answer is: {"columns": ["Number of trees sold", "Sales"], "methods": ["Mood Variance Test", "Levene Test", "Bartlett Test", "F-Test for Variance"]}'''


COT_DS_EG = '''# Column Information: 
column_header: Type of tree; data_type: categorical; num_of_rows: 14; is_normality: False.
column_header: Number of trees sold; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Average Tree Price; data_type: quantitative; num_of_rows: 14; is_normality: True.
column_header: Sales; data_type: quantitative; num_of_rows: 14; is_normality: True.
# Statistical Question: Could you determine what is the median value of the sold trees?
# Correct Answer: Firstly, the question asks about the sold trees, so the relevant column is Number of trees sold. Secondly, the purpose of the question is to calculate the median value, and the data of the relevant column is quantitative, so the applicable method is Median. Hence, the answer is: {"columns": ["Number of trees sold"], "methods": ["Median"]}'''


'''------------------
Stats prompt
'''
STATS_PROMPT = '''Methods and applicable usage scenarios:
# Correlation Analysis
Pearson Correlation Coefficient, Spearman Correlation Coefficient: Correlation analysis for two quantitative variables;
Kendall Correlation Coefficient: Correlation analysis for two quantitative variables, suitable for small samples;
Partial Correlation Coefficient: Correlation analysis when involving controlling variable;
# Distribution Compliance Test
Anderson-Darling Test, Kolmogorov-Smirnov Test for Normality: Test for normality;
Shapiro-Wilk Test of Normality: Test for normality, suitable for small samples;
Lilliefors Test: Test for normality, suitable for large samples;
Kolmogorov-Smirnov Test: Comparison of distribution between two independent samples;
Kolmogorov-Smirnov Test for Uniform distribution, Kolmogorov-Smirnov Test for Gamma distribution, Kolmogorov-Smirnov Test for Exponential distribution: Test for corresponding distributions;
# Contingency Table Test
Chi-square Independence Test: Contingency table test of large sample categorical variables;
Fisher Exact Test: Contingency table test of small sample categorical variables;
Mantel-Haenszel Test: Contingency table test when strata data to be controlled; 
# Variance Test
Mood Variance Test, Levene Test: Whether there is a significant difference;
Bartlett Test, F-Test for Variance: Whether there is a significant difference in variance between normally distributed variables;
# Descriptive Statistics
Mean, Median, Mode, Range, Quartile, Standard Deviation, Skewness, Kurtosis.'''

STRESSQA_STATS_PROMPT = '''Methods and applicable usage scenarios:
# Correlation Analysis
Pearson Correlation Coefficient, Spearman Correlation Coefficient: Correlation analysis for two quantitative variables;
Kendall Correlation Coefficient: Correlation analysis for two quantitative variables, suitable for small samples;
Partial Correlation Coefficient: Correlation analysis when involving controlling variable;
# Distribution Compliance Test
Anderson-Darling Test, Kolmogorov-Smirnov Test for Normality: Test for normality;
Shapiro-Wilk Test of Normality: Test for normality, suitable for small samples;
Lilliefors Test: Test for normality, suitable for large samples;
Kolmogorov-Smirnov Test: Comparison of distribution between two independent samples;
Kolmogorov-Smirnov Test for Uniform distribution, Kolmogorov-Smirnov Test for Gamma distribution, Kolmogorov-Smirnov Test for Exponential distribution: Test for corresponding distributions;
# Contingency Table Test
Chi-square Independence Test: Contingency table test of large sample categorical variables;
Fisher Exact Test: Contingency table test of small sample categorical variables;
Mantel-Haenszel Test: Contingency table test when strata data to be controlled; 
# Variance Test
Mood Variance Test, Levene Test: Whether there is a significant difference;
Bartlett Test, F-Test for Variance: Whether there is a significant difference in variance between normally distributed variables;
# Descriptive Statistics
Mean, Median, Mode, Range, Quartile, Standard Deviation, Skewness, Kurtosis;
# Group Comparison
Independent Samples t-test, Welch t-test: Compare means of two independent groups (use Welch when variances unequal);
Mann-Whitney U Test: Non-parametric alternative for two independent groups;
Paired Samples t-test, Wilcoxon Signed-Rank Test: Compare means of two related/paired groups;
One-Way ANOVA: Compare means across three or more independent groups (requires post-hoc if significant);
Kruskal-Wallis H Test: Non-parametric alternative for three or more groups;
# Regression
Simple Linear Regression: Model linear relationship between one predictor and outcome;
Multiple Linear Regression: Model linear relationship with multiple predictors;
Analysis of Covariance (ANCOVA): Compare group means while controlling for covariates;
# Multiple Testing
Multiple Endpoints with Benjamini-Hochberg: FDR correction for multiple comparisons (less conservative);
Multiple Endpoints with Bonferroni: Conservative correction for multiple comparisons;
Multiple Endpoints with Holm-Bonferroni: Step-down Bonferroni correction.'''
---
title: A comparison of results across multiple models.
toc-title: Table of contents
---

# Standard Imports + Data Listings

::: {.cell execution_count="1"}
``` {.python .cell-code}
import pandas as pd
import numpy as np

import os

from scipy import stats
from scipy.spatial.distance import jensenshannon

from IPython.display import Markdown, display

file_list = []

# Updated files following adjustment to api call available here:
# https://github.com/adamrossnelson/SurveyResponderMemos/tree/main/memos/2026-09-30-temp-sweep-wider/data
# Details on api call update: https://github.com/adamrossnelson/SurveyResponder/issues/18
file_list += [
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t0.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t0.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t0.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t0.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t1.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t1.2.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t1.4.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t1.6.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t1.8.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/falcon3_1b_t2.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t0.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t0.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t0.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t0.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.2.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.4.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.6.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.7.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.8.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t1.9.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite3.1-moe_1b_t2.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t0.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t0.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t0.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t0.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t1.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t1.2.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t1.4.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t1.6.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t1.8.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_350m_t2.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t0.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t0.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t0.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t0.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t1.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t1.2.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t1.4.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t1.6.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t1.8.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/granite4_3b_t2.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/lfm2_24b_t0.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/lfm2_24b_t0.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/lfm2_24b_t0.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/lfm2_24b_t1.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/lfm2_24b_t1.2.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/lfm2_24b_t1.4.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/lfm2_24b_t1.6.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/lfm2_24b_t1.8.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/lfm2_24b_t2.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t0.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t0.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t0.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t0.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t1.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t1.2.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t1.4.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t1.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t1.6.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t1.7.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t1.8.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t1.9.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/llama3.2_1b_t2.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t0.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t0.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t0.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t0.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t1.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t1.2.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t1.4.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t1.6.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t1.8.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/nemotron-mini_4b_t2.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t0.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t0.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t0.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t0.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t1.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t1.2.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t1.4.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t1.6.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t1.8.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/olmo2_7b_t2.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t0.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t0.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t0.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t0.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.0.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.2.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.25.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.4.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.5.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.6.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.7.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.75.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.8.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t1.9.csv",
    "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/temp-sweep-wider/qwen2.5-coder_0.5b_t2.0.csv"
]

# These files we assume were generated at the default model temperatures.
# file_list = [
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/GPT-OSS_20B_temp_1.50.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/GPT-OSS_20B_temp_1.65.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/GPT-OSS_20B_temp_1.75.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/GPT-OSS_20B_temp_1.85.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/PRCA_qwen2.5_7b_results_tempOf_1.65.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/PRCA_qwen2.5_7b_results_tempOf_1.75.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/PRCA_qwen2.5_7b_results_tempOf_1.85.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/PRCA_qwen2.5_7b_results_tempOf_2.00.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/prca_results_llama3.2_3b_temp1.65_cleaned.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/prca_results_mistral-nemo_12b_temp1.65_cleaned.csv",
#     "https://raw.githubusercontent.com/adamrossnelson/SurveyResponderData/refs/heads/main/data/newGen/prca_results_qwen3.6_27b_temp1.65_cleaned.csv",
# ]
```
:::

# Build Table

::::: {.cell execution_count="2"}
``` {.python .cell-code}
prca_cols = ['prca01','prca02','prca03','prca04','prca05','prca06','prca13','prca14','prca15','prca16','prca17','prca18']
group_cols = prca_cols[:6]  # Specify first six as the group ca columns
inter_cols = prca_cols[6:]  # Specify last six as the interpersonal ca columns

# Each scale we report on, mapped to the item columns that make it up.
scales = {
    'Group CA': group_cols,
    'Interpersonal CA': inter_cols,
    'Both Group and Interpersonal CA': prca_cols,
}

def ttest_vs_human(llm_totals, human_totals):
    """Welch's t-test (unequal variances) of LLM scale totals vs human scale totals."""
    if len(llm_totals) < 2 or len(human_totals) < 2:
        return 'N/A (insufficient observations)'
    if llm_totals.nunique() == 1 and human_totals.nunique() == 1:
        return 'N/A (no variance)'
    t_stat, p_val = stats.ttest_ind(llm_totals, human_totals, equal_var=False)
    return f't={t_stat:.2f}, p={p_val:.3f}'

def effect_vs_human(llm_totals, human_totals):
    """Mean difference (LLM - human) and Cohen's d using the pooled standard deviation."""
    n1, n2 = len(llm_totals), len(human_totals)
    if n1 < 2 or n2 < 2:
        return 'N/A (insufficient observations)'
    diff = llm_totals.mean() - human_totals.mean()
    pooled_sd = np.sqrt(((n1 - 1) * llm_totals.var(ddof=1) + (n2 - 1) * human_totals.var(ddof=1)) / (n1 + n2 - 2))
    d_text = 'N/A (zero pooled SD)' if np.isclose(pooled_sd, 0) else f'{diff / pooled_sd:.2f}'
    return f'Mean diff={diff:.2f} <br> Cohen\'s d={d_text}'

response_levels = [1, 2, 3, 4, 5]

def level_proportions(data):
    """Proportion of item responses at each level (1-5), pooled across the items in `data`."""
    values = np.asarray(data, dtype=float).ravel()
    counts = np.array([(values == level).sum() for level in response_levels], dtype=float)
    return counts / counts.sum() if counts.sum() > 0 else None

def distribution_vs_human(llm_props, human_props):
    """Similarity of the LLM and human response-level distributions (0 = identical, 1 = maximally different).

    Total variation distance = 0.5 * sum|p - q|; Jensen-Shannon distance uses log base 2.
    Wasserstein (earth mover's) distance treats levels as ordered; it is in response-level units (0 to 4).
    """
    if llm_props is None or human_props is None:
        return 'N/A (no valid responses)'
    tvd = 0.5 * np.abs(llm_props - human_props).sum()
    js = jensenshannon(llm_props, human_props, base=2)
    wd = stats.wasserstein_distance(response_levels, response_levels, llm_props, human_props)
    return f'Distribution distance vs human: <br> TVD={tvd:.2f}, JS={js:.2f}, WD={wd:.2f}'

def format_cell(
    mean, std, min_val, max_val, a_raw, a_std, prop_validated, t_test=None, effect=None, dist=None):
    """Render the mean/SD/alpha summary shown in a single table cell."""
    raw_text = a_raw if isinstance(a_raw, str) else f'{a_raw:.2f}'
    std_text = a_std if isinstance(a_std, str) else f'{a_std:.2f}'
    text = (
        f'Mean (SD): {mean:.2f} ({std:.2f}) <br><br> '
        f'min/max {min_val:.2f} / {max_val:.2f} <br><br> '
        f'α raw={raw_text} <br> α std={std_text} <br><br>'
        f'Prop Validated: {prop_validated:.2f}'
    )
    if t_test is not None:
        text += f'<br><br>t-test vs human: {t_test}'
    if effect is not None:
        text += f'<br><br>{effect}'
    if dist is not None:
        text += f'<br><br>{dist}'
    return text

def alpha_standardized(corr, data):
    """Standardized alpha from a correlation matrix. == Stata `alpha ..., std`."""
    R = np.asarray(corr, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("corr must be a square correlation matrix")
    k = R.shape[0]
    if k < 2:
        raise ValueError("need at least 2 items")
    if len(data) < 2:
        return 'N/A (fewer than 2 observations)'
    constant_cols = data.columns[data.nunique(dropna=True).eq(1)]
    if len(constant_cols):
        # return f'N/A (constant columns: {", ".join(map(str, constant_cols))})'
        return 'N/A (constant cols'
    if not np.isfinite(R).all():
        return 'N/A (undefined correlations)'
    r_bar = R[~np.eye(k, dtype=bool)].mean()
    denominator = 1 + (k - 1) * r_bar
    if np.isclose(denominator, 0):
        return 'N/A (zero standardized total variance)'
    return (k * r_bar) / denominator


def alpha_raw(df, ddof=1):
    """Raw (covariance-based) alpha from the item data. == Stata `alpha ...`.
 
    alpha = k/(k-1) * (1 - sum(item variances) / variance of the total)
    """
    data = pd.DataFrame(df)
    k = data.shape[1]
    if k < 2:
        raise ValueError("need at least 2 items")
    if len(data) < 2 or len(data) <= ddof:
        return 'N/A (insufficient observations)'
    constant_cols = data.columns[data.nunique(dropna=True).eq(1)]
    if len(constant_cols):
        # return f'N/A (constant columns: {", ".join(map(str, constant_cols))})'
        return 'N/A (constant cols)'
    S = np.asarray(data.cov(ddof=ddof), dtype=float)
    if not np.isfinite(S).all():
        return 'N/A (undefined covariance)'
    if np.isclose(S.sum(), 0):
        return 'N/A (zero total variance)'
    return (k / (k - 1)) * (1 - np.trace(S) / S.sum())

# Use os library to create data folder if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Bring data local for faster renders
for file in file_list:
    # Check if file is local in the data folder
    if not os.path.exists(f"data/{file.split('/')[-1]}"):
        # If not local download the file
        df = pd.read_csv(file, engine='python', on_bad_lines='skip')
        # Save the file locally
        df.to_csv(f"data/{file.split('/')[-1]}", index=False)

# Instantiate empty results list
results = []

df = pd.read_csv("data_human/human_responses.csv")
to_reverse = ['prca02', 'prca04', 'prca06', 'prca14', 'prca16', 'prca17']
df[to_reverse] = 6 - df[to_reverse]
df = df.dropna()
row = {'file': 'human_responses.csv'}
human_totals = {}
human_props = {}

for label, cols in scales.items():
    scale_total = df[cols].sum(axis=1)
    human_totals[label] = scale_total
    human_props[label] = level_proportions(df[cols])

    mean = scale_total.mean()
    std = scale_total.std()
    min_val = scale_total.min()
    max_val = scale_total.max()

    a_raw = alpha_raw(df[cols])
    a_std = alpha_standardized(df[cols].corr(), df[cols])

    prop_validated = 1.0

    row[label] = format_cell(
        mean, std, min_val, max_val, a_raw, a_std, prop_validated)

results.append(row)

for file in file_list:
    # Extract file name only from full path in file_list
    file_name = file.split('/')[-1]

    # Check if file is local
    if os.path.exists(f"data/{file_name}"):
        # Load the data file
        df = pd.read_csv(f"data/{file_name}", engine='python', on_bad_lines='skip')
    else:
        # Download the file
        df = pd.read_csv(file, engine='python', on_bad_lines='skip')

    # Track the starting number of observations
    starting_obs = len(df)

    # Remove observations (rows) that contained non-numeric values
    df[prca_cols] = df[prca_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=prca_cols)

    # Track the ending number of observations
    ending_obs = len(df)

    # Check for values outside 0-5 range
    if df[prca_cols][df[prca_cols] > 5].sum().sum() > 0 or df[prca_cols][df[prca_cols] < 0].sum().sum() > 0:
        print(f"File: {file_name} (WARNING: Values outside 0-5 range detected)")
        print(f"Observations with values > 5: {df[prca_cols][df[prca_cols] > 5].sum().sum()}")
        print(f"Observations with values < 0: {df[prca_cols][df[prca_cols] < 0].sum().sum()}")

    # Build one cell per scale (Group CA, Interpersonal CA, Both)
    row = {'file': file.split('/')[-1]}

    for label, cols in scales.items():
        scale_total = df[cols].sum(axis=1)

        mean = scale_total.mean()
        std = scale_total.std()
        min_val = scale_total.min()
        max_val = scale_total.max()

        a_raw = alpha_raw(df[cols])
        a_std = alpha_standardized(df[cols].corr(), df[cols])

        prop_validated = ending_obs / starting_obs

        t_test = ttest_vs_human(scale_total, human_totals[label])
        effect = effect_vs_human(scale_total, human_totals[label])
        dist = distribution_vs_human(level_proportions(df[cols]), human_props[label])

        row[label] = format_cell(
            mean, std, min_val, max_val, a_raw, a_std, prop_validated, t_test, effect, dist)

    results.append(row)

summary_table = pd.DataFrame(results)
display(Markdown(summary_table.to_markdown()))
```

::: {.cell-output .cell-output-stderr}
    /opt/anaconda3/lib/python3.12/site-packages/scipy/stats/_axis_nan_policy.py:531: RuntimeWarning:

    Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.

    /opt/anaconda3/lib/python3.12/site-packages/scipy/stats/_axis_nan_policy.py:531: RuntimeWarning:

    Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
:::

::: {.cell-output .cell-output-display .cell-output-markdown}
  ---------------------------------------------------------------------------------------------------------------------------------------------------------
        file                           Group CA                               Interpersonal CA                       Both Group and Interpersonal CA
  ----- ------------------------------ -------------------------------------- -------------------------------------- --------------------------------------
      0 human_responses.csv            Mean (SD): 14.89 (6.03)                Mean (SD): 14.68 (5.95)                Mean (SD): 29.58 (11.51)
                                       `<br><br>`{=html} min/max 6.00 / 30.00 `<br><br>`{=html} min/max 6.00 / 30.00 `<br><br>`{=html} min/max 12.00 /
                                       `<br><br>`{=html} α raw=0.92           `<br><br>`{=html} α raw=0.92           60.00 `<br><br>`{=html} α raw=0.95
                                       `<br>`{=html} α std=0.92               `<br>`{=html} α std=0.92               `<br>`{=html} α std=0.95
                                       `<br><br>`{=html}Prop Validated: 1.00  `<br><br>`{=html}Prop Validated: 1.00  `<br><br>`{=html}Prop Validated: 1.00

      1 falcon3_1b_t0.0.csv            Mean (SD): 17.92 (0.80)                Mean (SD): 18.02 (0.14)                Mean (SD): 35.94 (0.82)
                                       `<br><br>`{=html} min/max 16.00 /      `<br><br>`{=html} min/max 18.00 /      `<br><br>`{=html} min/max 34.00 /
                                       20.00 `<br><br>`{=html} α raw=N/A      19.00 `<br><br>`{=html} α raw=N/A      38.00 `<br><br>`{=html} α raw=N/A
                                       (constant cols) `<br>`{=html} α        (constant cols) `<br>`{=html} α        (constant cols) `<br>`{=html} α
                                       std=N/A (constant cols                 std=N/A (constant cols                 std=N/A (constant cols
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.50`<br><br>`{=html}t-test vs human:  0.50`<br><br>`{=html}t-test vs human:  0.50`<br><br>`{=html}t-test vs human:
                                       t=7.75, p=0.000`<br><br>`{=html}Mean   t=9.02, p=0.000`<br><br>`{=html}Mean   t=8.80, p=0.000`<br><br>`{=html}Mean
                                       diff=3.03 `<br>`{=html} Cohen's        diff=3.34 `<br>`{=html} Cohen's        diff=6.36 `<br>`{=html} Cohen's
                                       d=0.55`<br><br>`{=html}Distribution    d=0.61`<br><br>`{=html}Distribution    d=0.60`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.84, JS=0.80, WD=1.10             TVD=0.87, JS=0.83, WD=1.16             TVD=0.85, JS=0.81, WD=1.13

      2 falcon3_1b_t0.25.csv           Mean (SD): 18.24 (1.62)                Mean (SD): 18.14 (0.96)                Mean (SD): 36.37 (2.14)
                                       `<br><br>`{=html} min/max 14.00 /      `<br><br>`{=html} min/max 15.00 /      `<br><br>`{=html} min/max 32.00 /
                                       22.00 `<br><br>`{=html} α raw=0.20     22.00 `<br><br>`{=html} α raw=0.33     43.00 `<br><br>`{=html} α raw=0.42
                                       `<br>`{=html} α std=0.26               `<br>`{=html} α std=0.27               `<br>`{=html} α std=0.40
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.51`<br><br>`{=html}t-test vs human:  0.51`<br><br>`{=html}t-test vs human:  0.51`<br><br>`{=html}t-test vs human:
                                       t=7.65, p=0.000`<br><br>`{=html}Mean   t=8.79, p=0.000`<br><br>`{=html}Mean   t=8.78, p=0.000`<br><br>`{=html}Mean
                                       diff=3.34 `<br>`{=html} Cohen's        diff=3.45 `<br>`{=html} Cohen's        diff=6.80 `<br>`{=html} Cohen's
                                       d=0.60`<br><br>`{=html}Distribution    d=0.63`<br><br>`{=html}Distribution    d=0.64`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.72, JS=0.66, WD=0.93             TVD=0.80, JS=0.75, WD=1.08             TVD=0.76, JS=0.70, WD=1.00

      3 falcon3_1b_t0.5.csv            Mean (SD): 18.39 (1.65)                Mean (SD): 18.29 (1.13)                Mean (SD): 36.68 (1.90)
                                       `<br><br>`{=html} min/max 16.00 /      `<br><br>`{=html} min/max 16.00 /      `<br><br>`{=html} min/max 33.00 /
                                       22.00 `<br><br>`{=html} α raw=-0.33    21.00 `<br><br>`{=html} α raw=-0.83    41.00 `<br><br>`{=html} α raw=-0.60
                                       `<br>`{=html} α std=-0.51              `<br>`{=html} α std=-0.66              `<br>`{=html} α std=-1.00
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.31`<br><br>`{=html}t-test vs human:  0.31`<br><br>`{=html}t-test vs human:  0.31`<br><br>`{=html}t-test vs human:
                                       t=7.33, p=0.000`<br><br>`{=html}Mean   t=8.56, p=0.000`<br><br>`{=html}Mean   t=8.97, p=0.000`<br><br>`{=html}Mean
                                       diff=3.49 `<br>`{=html} Cohen's        diff=3.61 `<br>`{=html} Cohen's        diff=7.10 `<br>`{=html} Cohen's
                                       d=0.61`<br><br>`{=html}Distribution    d=0.64`<br><br>`{=html}Distribution    d=0.65`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.61, JS=0.55, WD=0.78             TVD=0.66, JS=0.61, WD=0.90             TVD=0.62, JS=0.58, WD=0.82

      4 falcon3_1b_t0.75.csv           Mean (SD): 18.28 (1.87)                Mean (SD): 18.78 (2.41)                Mean (SD): 37.06 (3.24)
                                       `<br><br>`{=html} min/max 15.00 /      `<br><br>`{=html} min/max 15.00 /      `<br><br>`{=html} min/max 30.00 /
                                       23.00 `<br><br>`{=html} α raw=-0.03    23.00 `<br><br>`{=html} α raw=0.44     43.00 `<br><br>`{=html} α raw=0.34
                                       `<br>`{=html} α std=-0.31              `<br>`{=html} α std=0.45               `<br>`{=html} α std=0.24
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.18`<br><br>`{=html}t-test vs human:  0.18`<br><br>`{=html}t-test vs human:  0.18`<br><br>`{=html}t-test vs human:
                                       t=5.85, p=0.000`<br><br>`{=html}Mean   t=6.03, p=0.000`<br><br>`{=html}Mean   t=7.15, p=0.000`<br><br>`{=html}Mean
                                       diff=3.39 `<br>`{=html} Cohen's        diff=4.09 `<br>`{=html} Cohen's        diff=7.48 `<br>`{=html} Cohen's
                                       d=0.58`<br><br>`{=html}Distribution    d=0.71`<br><br>`{=html}Distribution    d=0.67`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.50, JS=0.47, WD=0.68             TVD=0.53, JS=0.51, WD=0.72             TVD=0.51, JS=0.49, WD=0.70

      5 falcon3_1b_t1.0.csv            Mean (SD): 19.08 (2.02)                Mean (SD): 17.42 (2.78)                Mean (SD): 36.50 (3.87)
                                       `<br><br>`{=html} min/max 15.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 28.00 /
                                       22.00 `<br><br>`{=html} α raw=-0.03    23.00 `<br><br>`{=html} α raw=0.29     43.00 `<br><br>`{=html} α raw=0.36
                                       `<br>`{=html} α std=-0.09              `<br>`{=html} α std=0.34               `<br>`{=html} α std=0.43
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.12`<br><br>`{=html}t-test vs human:  0.12`<br><br>`{=html}t-test vs human:  0.12`<br><br>`{=html}t-test vs human:
                                       t=6.05, p=0.000`<br><br>`{=html}Mean   t=3.09, p=0.007`<br><br>`{=html}Mean   t=5.22, p=0.000`<br><br>`{=html}Mean
                                       diff=4.19 `<br>`{=html} Cohen's        diff=2.73 `<br>`{=html} Cohen's        diff=6.92 `<br>`{=html} Cohen's
                                       d=0.71`<br><br>`{=html}Distribution    d=0.47`<br><br>`{=html}Distribution    d=0.61`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.44, JS=0.42, WD=0.70             TVD=0.48, JS=0.43, WD=0.59             TVD=0.46, JS=0.41, WD=0.61

      6 falcon3_1b_t1.2.csv            Mean (SD): 17.07 (2.05)                Mean (SD): 18.93 (1.79)                Mean (SD): 36.00 (2.80)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 15.00 /      `<br><br>`{=html} min/max 31.00 /
                                       19.00 `<br><br>`{=html} α raw=-0.26    22.00 `<br><br>`{=html} α raw=-1.00    41.00 `<br><br>`{=html} α raw=-0.43
                                       `<br>`{=html} α std=-0.22              `<br>`{=html} α std=-1.03              `<br>`{=html} α std=-0.46
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.15`<br><br>`{=html}t-test vs human:  0.15`<br><br>`{=html}t-test vs human:  0.15`<br><br>`{=html}t-test vs human:
                                       t=3.35, p=0.002`<br><br>`{=html}Mean   t=7.18, p=0.000`<br><br>`{=html}Mean   t=6.32, p=0.000`<br><br>`{=html}Mean
                                       diff=2.17 `<br>`{=html} Cohen's        diff=4.25 `<br>`{=html} Cohen's        diff=6.42 `<br>`{=html} Cohen's
                                       d=0.37`<br><br>`{=html}Distribution    d=0.73`<br><br>`{=html}Distribution    d=0.57`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.48, JS=0.46, WD=0.56             TVD=0.37, JS=0.36, WD=0.71             TVD=0.42, JS=0.39, WD=0.55

      7 falcon3_1b_t1.4.csv            Mean (SD): 16.67 (2.18)                Mean (SD): 18.89 (2.57)                Mean (SD): 35.56 (3.47)
                                       `<br><br>`{=html} min/max 14.00 /      `<br><br>`{=html} min/max 15.00 /      `<br><br>`{=html} min/max 32.00 /
                                       20.00 `<br><br>`{=html} α raw=-1.12    24.00 `<br><br>`{=html} α raw=0.30     40.00 `<br><br>`{=html} α raw=-0.19
                                       `<br>`{=html} α std=-1.08              `<br>`{=html} α std=0.30               `<br>`{=html} α std=0.08
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.09`<br><br>`{=html}t-test vs human:  0.09`<br><br>`{=html}t-test vs human:  0.09`<br><br>`{=html}t-test vs human:
                                       t=2.17, p=0.049`<br><br>`{=html}Mean   t=4.51, p=0.001`<br><br>`{=html}Mean   t=4.40, p=0.001`<br><br>`{=html}Mean
                                       diff=1.77 `<br>`{=html} Cohen's        diff=4.20 `<br>`{=html} Cohen's        diff=5.98 `<br>`{=html} Cohen's
                                       d=0.30`<br><br>`{=html}Distribution    d=0.72`<br><br>`{=html}Distribution    d=0.53`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.32, JS=0.34, WD=0.32             TVD=0.48, JS=0.45, WD=0.70             TVD=0.40, JS=0.38, WD=0.50

      8 falcon3_1b_t1.6.csv            Mean (SD): 17.29 (2.09)                Mean (SD): 18.29 (2.49)                Mean (SD): 35.57 (3.86)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 14.00 /      `<br><br>`{=html} min/max 29.00 /
                                       20.00 `<br><br>`{=html} α raw=-0.73    22.00 `<br><br>`{=html} α raw=-0.21    41.00 `<br><br>`{=html} α raw=0.04
                                       `<br>`{=html} α std=-1.00              `<br>`{=html} α std=-0.10              `<br>`{=html} α std=0.00
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.14`<br><br>`{=html}t-test vs human:  0.14`<br><br>`{=html}t-test vs human:  0.14`<br><br>`{=html}t-test vs human:
                                       t=3.56, p=0.001`<br><br>`{=html}Mean   t=4.73, p=0.000`<br><br>`{=html}Mean   t=4.78, p=0.000`<br><br>`{=html}Mean
                                       diff=2.39 `<br>`{=html} Cohen's        diff=3.60 `<br>`{=html} Cohen's        diff=5.99 `<br>`{=html} Cohen's
                                       d=0.41`<br><br>`{=html}Distribution    d=0.62`<br><br>`{=html}Distribution    d=0.53`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.34, JS=0.33, WD=0.40             TVD=0.40, JS=0.35, WD=0.60             TVD=0.37, JS=0.34, WD=0.50

      9 falcon3_1b_t1.8.csv            Mean (SD): 19.50 (1.86)                Mean (SD): 17.72 (2.49)                Mean (SD): 37.22 (3.61)
                                       `<br><br>`{=html} min/max 17.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 31.00 /
                                       23.00 `<br><br>`{=html} α raw=-1.08    23.00 `<br><br>`{=html} α raw=-0.08    45.00 `<br><br>`{=html} α raw=-0.02
                                       `<br>`{=html} α std=-1.14              `<br>`{=html} α std=-0.36              `<br>`{=html} α std=-0.30
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.18`<br><br>`{=html}t-test vs human:  0.18`<br><br>`{=html}t-test vs human:  0.18`<br><br>`{=html}t-test vs human:
                                       t=8.01, p=0.000`<br><br>`{=html}Mean   t=4.38, p=0.000`<br><br>`{=html}Mean   t=6.89, p=0.000`<br><br>`{=html}Mean
                                       diff=4.61 `<br>`{=html} Cohen's        diff=3.04 `<br>`{=html} Cohen's        diff=7.65 `<br>`{=html} Cohen's
                                       d=0.79`<br><br>`{=html}Distribution    d=0.52`<br><br>`{=html}Distribution    d=0.68`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.41, JS=0.39, WD=0.77             TVD=0.40, JS=0.35, WD=0.52             TVD=0.40, JS=0.37, WD=0.64

     10 falcon3_1b_t2.0.csv            Mean (SD): 17.64 (2.80)                Mean (SD): 18.36 (2.56)                Mean (SD): 36.00 (3.61)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 29.00 /
                                       22.00 `<br><br>`{=html} α raw=0.02     23.00 `<br><br>`{=html} α raw=0.14     42.00 `<br><br>`{=html} α raw=-0.04
                                       `<br>`{=html} α std=0.07               `<br>`{=html} α std=0.10               `<br>`{=html} α std=-0.06
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.22`<br><br>`{=html}t-test vs human:  0.22`<br><br>`{=html}t-test vs human:  0.22`<br><br>`{=html}t-test vs human:
                                       t=3.89, p=0.000`<br><br>`{=html}Mean   t=5.59, p=0.000`<br><br>`{=html}Mean   t=6.12, p=0.000`<br><br>`{=html}Mean
                                       diff=2.74 `<br>`{=html} Cohen's        diff=3.68 `<br>`{=html} Cohen's        diff=6.42 `<br>`{=html} Cohen's
                                       d=0.47`<br><br>`{=html}Distribution    d=0.64`<br><br>`{=html}Distribution    d=0.58`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.34, JS=0.32, WD=0.46             TVD=0.45, JS=0.42, WD=0.61             TVD=0.40, JS=0.37, WD=0.54

     11 granite3.1-moe_1b_t0.0.csv     Mean (SD): 18.04 (1.61)                Mean (SD): 18.00 (1.58)                Mean (SD): 36.04 (2.22)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 14.00 /      `<br><br>`{=html} min/max 28.00 /
                                       22.00 `<br><br>`{=html} α raw=0.05     23.00 `<br><br>`{=html} α raw=-0.01    44.00 `<br><br>`{=html} α raw=-0.01
                                       `<br>`{=html} α std=0.06               `<br>`{=html} α std=-0.00              `<br>`{=html} α std=-0.02
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.17, p=0.000`<br><br>`{=html}Mean   t=8.71, p=0.000`<br><br>`{=html}Mean   t=8.91, p=0.000`<br><br>`{=html}Mean
                                       diff=3.15 `<br>`{=html} Cohen's        diff=3.31 `<br>`{=html} Cohen's        diff=6.46 `<br>`{=html} Cohen's
                                       d=0.74`<br><br>`{=html}Distribution    d=0.78`<br><br>`{=html}Distribution    d=0.81`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.24, JS=0.29, WD=0.54             TVD=0.24, JS=0.29, WD=0.56             TVD=0.24, JS=0.29, WD=0.55

     12 granite3.1-moe_1b_t0.25.csv    Mean (SD): 18.09 (1.20)                Mean (SD): 18.06 (1.20)                Mean (SD): 36.15 (1.74)
                                       `<br><br>`{=html} min/max 15.00 /      `<br><br>`{=html} min/max 15.00 /      `<br><br>`{=html} min/max 31.00 /
                                       22.00 `<br><br>`{=html} α raw=-0.15    22.00 `<br><br>`{=html} α raw=0.04     42.00 `<br><br>`{=html} α raw=-0.00
                                       `<br>`{=html} α std=-0.15              `<br>`{=html} α std=0.04               `<br>`{=html} α std=0.01
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.42, p=0.000`<br><br>`{=html}Mean   t=8.98, p=0.000`<br><br>`{=html}Mean   t=9.12, p=0.000`<br><br>`{=html}Mean
                                       diff=3.20 `<br>`{=html} Cohen's        diff=3.37 `<br>`{=html} Cohen's        diff=6.57 `<br>`{=html} Cohen's
                                       d=0.76`<br><br>`{=html}Distribution    d=0.81`<br><br>`{=html}Distribution    d=0.83`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.26, JS=0.30, WD=0.53             TVD=0.26, JS=0.31, WD=0.56             TVD=0.26, JS=0.30, WD=0.55

     13 granite3.1-moe_1b_t0.5.csv     Mean (SD): 17.93 (1.33)                Mean (SD): 17.98 (1.56)                Mean (SD): 35.91 (2.06)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 27.00 /
                                       25.00 `<br><br>`{=html} α raw=-0.13    23.00 `<br><br>`{=html} α raw=0.05     42.00 `<br><br>`{=html} α raw=-0.01
                                       `<br>`{=html} α std=-0.10              `<br>`{=html} α std=0.03               `<br>`{=html} α std=-0.01
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.95, p=0.000`<br><br>`{=html}Mean   t=8.68, p=0.000`<br><br>`{=html}Mean   t=8.75, p=0.000`<br><br>`{=html}Mean
                                       diff=3.03 `<br>`{=html} Cohen's        diff=3.30 `<br>`{=html} Cohen's        diff=6.33 `<br>`{=html} Cohen's
                                       d=0.72`<br><br>`{=html}Distribution    d=0.78`<br><br>`{=html}Distribution    d=0.79`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.24, JS=0.27, WD=0.51             TVD=0.24, JS=0.27, WD=0.55             TVD=0.24, JS=0.27, WD=0.53

     14 granite3.1-moe_1b_t0.75.csv    Mean (SD): 18.07 (1.58)                Mean (SD): 18.12 (1.44)                Mean (SD): 36.19 (2.07)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 28.00 /
                                       22.00 `<br><br>`{=html} α raw=-0.05    22.00 `<br><br>`{=html} α raw=-0.17    41.00 `<br><br>`{=html} α raw=-0.17
                                       `<br>`{=html} α std=-0.07              `<br>`{=html} α std=-0.17              `<br>`{=html} α std=-0.20
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.26, p=0.000`<br><br>`{=html}Mean   t=9.08, p=0.000`<br><br>`{=html}Mean   t=9.14, p=0.000`<br><br>`{=html}Mean
                                       diff=3.18 `<br>`{=html} Cohen's        diff=3.44 `<br>`{=html} Cohen's        diff=6.61 `<br>`{=html} Cohen's
                                       d=0.74`<br><br>`{=html}Distribution    d=0.82`<br><br>`{=html}Distribution    d=0.83`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.26, WD=0.53             TVD=0.23, JS=0.26, WD=0.57             TVD=0.23, JS=0.26, WD=0.55

     15 granite3.1-moe_1b_t1.0.csv     Mean (SD): 18.18 (1.61)                Mean (SD): 18.22 (1.71)                Mean (SD): 36.40 (2.34)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 29.00 /
                                       24.00 `<br><br>`{=html} α raw=0.02     25.00 `<br><br>`{=html} α raw=-0.07    45.00 `<br><br>`{=html} α raw=-0.04
                                       `<br>`{=html} α std=0.01               `<br>`{=html} α std=-0.07              `<br>`{=html} α std=-0.04
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.99`<br><br>`{=html}t-test vs human:  0.99`<br><br>`{=html}t-test vs human:  0.99`<br><br>`{=html}t-test vs human:
                                       t=8.53, p=0.000`<br><br>`{=html}Mean   t=9.24, p=0.000`<br><br>`{=html}Mean   t=9.39, p=0.000`<br><br>`{=html}Mean
                                       diff=3.29 `<br>`{=html} Cohen's        diff=3.53 `<br>`{=html} Cohen's        diff=6.82 `<br>`{=html} Cohen's
                                       d=0.77`<br><br>`{=html}Distribution    d=0.83`<br><br>`{=html}Distribution    d=0.85`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.24, WD=0.55             TVD=0.23, JS=0.25, WD=0.59             TVD=0.23, JS=0.24, WD=0.57

     16 granite3.1-moe_1b_t1.2.csv     Mean (SD): 18.13 (1.64)                Mean (SD): 18.35 (1.66)                Mean (SD): 36.48 (2.24)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 27.00 /
                                       23.00 `<br><br>`{=html} α raw=-0.13    26.00 `<br><br>`{=html} α raw=-0.07    44.00 `<br><br>`{=html} α raw=-0.19
                                       `<br>`{=html} α std=-0.12              `<br>`{=html} α std=-0.06              `<br>`{=html} α std=-0.19
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.39, p=0.000`<br><br>`{=html}Mean   t=9.62, p=0.000`<br><br>`{=html}Mean   t=9.52, p=0.000`<br><br>`{=html}Mean
                                       diff=3.23 `<br>`{=html} Cohen's        diff=3.67 `<br>`{=html} Cohen's        diff=6.90 `<br>`{=html} Cohen's
                                       d=0.76`<br><br>`{=html}Distribution    d=0.87`<br><br>`{=html}Distribution    d=0.86`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.23, WD=0.54             TVD=0.24, JS=0.24, WD=0.61             TVD=0.23, JS=0.24, WD=0.58

     17 granite3.1-moe_1b_t1.25.csv    Mean (SD): 17.27 (2.43)                Mean (SD): 18.27 (1.57)                Mean (SD): 35.53 (3.12)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 25.00 /
                                       24.00 `<br><br>`{=html} α raw=0.38     20.00 `<br><br>`{=html} α raw=0.03     43.00 `<br><br>`{=html} α raw=0.36
                                       `<br>`{=html} α std=0.40               `<br>`{=html} α std=-0.01              `<br>`{=html} α std=0.35
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=4.09, p=0.000`<br><br>`{=html}Mean   t=7.66, p=0.000`<br><br>`{=html}Mean   t=6.53, p=0.000`<br><br>`{=html}Mean
                                       diff=2.37 `<br>`{=html} Cohen's        diff=3.58 `<br>`{=html} Cohen's        diff=5.96 `<br>`{=html} Cohen's
                                       d=0.41`<br><br>`{=html}Distribution    d=0.63`<br><br>`{=html}Distribution    d=0.54`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.20, JS=0.22, WD=0.40             TVD=0.22, JS=0.22, WD=0.60             TVD=0.21, JS=0.21, WD=0.50

     18 granite3.1-moe_1b_t1.4.csv     Mean (SD): 18.18 (1.89)                Mean (SD): 18.21 (1.76)                Mean (SD): 36.39 (2.57)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 28.00 /
                                       24.00 `<br><br>`{=html} α raw=0.00     24.00 `<br><br>`{=html} α raw=0.08     43.00 `<br><br>`{=html} α raw=0.03
                                       `<br>`{=html} α std=-0.00              `<br>`{=html} α std=0.09               `<br>`{=html} α std=0.04
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.46, p=0.000`<br><br>`{=html}Mean   t=9.20, p=0.000`<br><br>`{=html}Mean   t=9.35, p=0.000`<br><br>`{=html}Mean
                                       diff=3.29 `<br>`{=html} Cohen's        diff=3.52 `<br>`{=html} Cohen's        diff=6.81 `<br>`{=html} Cohen's
                                       d=0.76`<br><br>`{=html}Distribution    d=0.83`<br><br>`{=html}Distribution    d=0.85`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.23, JS=0.24, WD=0.55             TVD=0.24, JS=0.25, WD=0.59             TVD=0.23, JS=0.24, WD=0.57

     19 granite3.1-moe_1b_t1.5.csv     Mean (SD): 18.25 (1.55)                Mean (SD): 18.20 (1.78)                Mean (SD): 36.44 (2.44)
                                       `<br><br>`{=html} min/max 14.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 29.00 /
                                       23.00 `<br><br>`{=html} α raw=-0.19    24.00 `<br><br>`{=html} α raw=-0.00    47.00 `<br><br>`{=html} α raw=-0.00
                                       `<br>`{=html} α std=-0.18              `<br>`{=html} α std=-0.01              `<br>`{=html} α std=-0.01
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.73, p=0.000`<br><br>`{=html}Mean   t=9.17, p=0.000`<br><br>`{=html}Mean   t=9.44, p=0.000`<br><br>`{=html}Mean
                                       diff=3.36 `<br>`{=html} Cohen's        diff=3.51 `<br>`{=html} Cohen's        diff=6.87 `<br>`{=html} Cohen's
                                       d=0.79`<br><br>`{=html}Distribution    d=0.82`<br><br>`{=html}Distribution    d=0.85`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.24, WD=0.56             TVD=0.23, JS=0.24, WD=0.59             TVD=0.23, JS=0.24, WD=0.57

     20 granite3.1-moe_1b_t1.6.csv     Mean (SD): 18.31 (1.98)                Mean (SD): 18.23 (1.79)                Mean (SD): 36.54 (2.68)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 29.00 /
                                       25.00 `<br><br>`{=html} α raw=0.19     24.00 `<br><br>`{=html} α raw=0.07     43.00 `<br><br>`{=html} α raw=0.13
                                       `<br>`{=html} α std=0.19               `<br>`{=html} α std=0.07               `<br>`{=html} α std=0.12
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.75, p=0.000`<br><br>`{=html}Mean   t=9.24, p=0.000`<br><br>`{=html}Mean   t=9.54, p=0.000`<br><br>`{=html}Mean
                                       diff=3.42 `<br>`{=html} Cohen's        diff=3.54 `<br>`{=html} Cohen's        diff=6.96 `<br>`{=html} Cohen's
                                       d=0.79`<br><br>`{=html}Distribution    d=0.83`<br><br>`{=html}Distribution    d=0.86`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.23, JS=0.23, WD=0.57             TVD=0.23, JS=0.23, WD=0.59             TVD=0.23, JS=0.23, WD=0.58

     21 granite3.1-moe_1b_t1.7.csv     Mean (SD): 18.32 (1.76)                Mean (SD): 18.18 (1.77)                Mean (SD): 36.50 (2.42)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 30.00 /
                                       23.00 `<br><br>`{=html} α raw=-0.17    23.00 `<br><br>`{=html} α raw=-0.02    44.00 `<br><br>`{=html} α raw=-0.16
                                       `<br>`{=html} α std=-0.18              `<br>`{=html} α std=-0.01              `<br>`{=html} α std=-0.18
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.84, p=0.000`<br><br>`{=html}Mean   t=9.13, p=0.000`<br><br>`{=html}Mean   t=9.52, p=0.000`<br><br>`{=html}Mean
                                       diff=3.43 `<br>`{=html} Cohen's        diff=3.50 `<br>`{=html} Cohen's        diff=6.92 `<br>`{=html} Cohen's
                                       d=0.80`<br><br>`{=html}Distribution    d=0.82`<br><br>`{=html}Distribution    d=0.86`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.23, WD=0.57             TVD=0.22, JS=0.22, WD=0.58             TVD=0.22, JS=0.23, WD=0.58

     22 granite3.1-moe_1b_t1.75.csv    Mean (SD): 18.00 (1.93)                Mean (SD): 17.41 (2.46)                Mean (SD): 35.41 (3.39)
                                       `<br><br>`{=html} min/max 14.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 27.00 /
                                       23.00 `<br><br>`{=html} α raw=-0.01    22.00 `<br><br>`{=html} α raw=0.33     41.00 `<br><br>`{=html} α raw=0.32
                                       `<br>`{=html} α std=0.08               `<br>`{=html} α std=0.35               `<br>`{=html} α std=0.35
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.01, p=0.000`<br><br>`{=html}Mean   t=4.65, p=0.000`<br><br>`{=html}Mean   t=6.14, p=0.000`<br><br>`{=html}Mean
                                       diff=3.11 `<br>`{=html} Cohen's        diff=2.73 `<br>`{=html} Cohen's        diff=5.84 `<br>`{=html} Cohen's
                                       d=0.54`<br><br>`{=html}Distribution    d=0.48`<br><br>`{=html}Distribution    d=0.53`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.20, JS=0.22, WD=0.52             TVD=0.19, JS=0.21, WD=0.45             TVD=0.20, JS=0.22, WD=0.49

     23 granite3.1-moe_1b_t1.8.csv     Mean (SD): 18.18 (1.95)                Mean (SD): 18.18 (1.86)                Mean (SD): 36.35 (2.68)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 29.00 /
                                       24.00 `<br><br>`{=html} α raw=0.02     23.00 `<br><br>`{=html} α raw=-0.00    44.00 `<br><br>`{=html} α raw=-0.01
                                       `<br>`{=html} α std=0.02               `<br>`{=html} α std=0.02               `<br>`{=html} α std=0.00
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.42, p=0.000`<br><br>`{=html}Mean   t=9.08, p=0.000`<br><br>`{=html}Mean   t=9.28, p=0.000`<br><br>`{=html}Mean
                                       diff=3.28 `<br>`{=html} Cohen's        diff=3.49 `<br>`{=html} Cohen's        diff=6.78 `<br>`{=html} Cohen's
                                       d=0.76`<br><br>`{=html}Distribution    d=0.82`<br><br>`{=html}Distribution    d=0.84`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.22, WD=0.55             TVD=0.23, JS=0.23, WD=0.58             TVD=0.23, JS=0.23, WD=0.56

     24 granite3.1-moe_1b_t1.9.csv     Mean (SD): 17.97 (1.83)                Mean (SD): 18.26 (1.94)                Mean (SD): 36.24 (2.61)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 27.00 /
                                       23.00 `<br><br>`{=html} α raw=-0.13    24.00 `<br><br>`{=html} α raw=0.07     43.00 `<br><br>`{=html} α raw=-0.07
                                       `<br>`{=html} α std=-0.13              `<br>`{=html} α std=0.07               `<br>`{=html} α std=-0.06
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.93, p=0.000`<br><br>`{=html}Mean   t=9.28, p=0.000`<br><br>`{=html}Mean   t=9.13, p=0.000`<br><br>`{=html}Mean
                                       diff=3.08 `<br>`{=html} Cohen's        diff=3.58 `<br>`{=html} Cohen's        diff=6.66 `<br>`{=html} Cohen's
                                       d=0.71`<br><br>`{=html}Distribution    d=0.83`<br><br>`{=html}Distribution    d=0.82`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.22, WD=0.51             TVD=0.24, JS=0.23, WD=0.60             TVD=0.23, JS=0.23, WD=0.56

     25 granite3.1-moe_1b_t2.0.csv     Mean (SD): 18.44 (1.75)                Mean (SD): 18.28 (1.94)                Mean (SD): 36.72 (2.56)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 29.00 /
                                       24.00 `<br><br>`{=html} α raw=-0.12    25.00 `<br><br>`{=html} α raw=0.02     43.00 `<br><br>`{=html} α raw=-0.08
                                       `<br>`{=html} α std=-0.11              `<br>`{=html} α std=0.03               `<br>`{=html} α std=-0.07
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=9.16, p=0.000`<br><br>`{=html}Mean   t=9.31, p=0.000`<br><br>`{=html}Mean   t=9.80, p=0.000`<br><br>`{=html}Mean
                                       diff=3.55 `<br>`{=html} Cohen's        diff=3.59 `<br>`{=html} Cohen's        diff=7.14 `<br>`{=html} Cohen's
                                       d=0.82`<br><br>`{=html}Distribution    d=0.84`<br><br>`{=html}Distribution    d=0.88`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.23, JS=0.23, WD=0.59             TVD=0.23, JS=0.23, WD=0.60             TVD=0.23, JS=0.23, WD=0.60

     26 granite4_350m_t0.0.csv         Mean (SD): 17.26 (1.32)                Mean (SD): 17.70 (1.00)                Mean (SD): 34.96 (2.00)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 16.00 /      `<br><br>`{=html} min/max 28.00 /
                                       18.00 `<br><br>`{=html} α raw=N/A      20.00 `<br><br>`{=html} α raw=-3.34    38.00 `<br><br>`{=html} α raw=N/A
                                       (constant cols) `<br>`{=html} α        `<br>`{=html} α std=-4.14              (constant cols) `<br>`{=html} α
                                       std=N/A (constant cols                 `<br><br>`{=html}Prop Validated:       std=N/A (constant cols
                                       `<br><br>`{=html}Prop Validated:       1.00`<br><br>`{=html}t-test vs human:  `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  t=7.88, p=0.000`<br><br>`{=html}Mean   1.00`<br><br>`{=html}t-test vs human:
                                       t=5.97, p=0.000`<br><br>`{=html}Mean   diff=3.02 `<br>`{=html} Cohen's        t=7.26, p=0.000`<br><br>`{=html}Mean
                                       diff=2.37 `<br>`{=html} Cohen's        d=0.59`<br><br>`{=html}Distribution    diff=5.38 `<br>`{=html} Cohen's
                                       d=0.46`<br><br>`{=html}Distribution    distance vs human: `<br>`{=html}       d=0.55`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       TVD=0.58, JS=0.64, WD=0.77             distance vs human: `<br>`{=html}
                                       TVD=0.57, JS=0.64, WD=0.85                                                    TVD=0.57, JS=0.63, WD=0.81

     27 granite4_350m_t0.25.csv        Mean (SD): 17.45 (1.95)                Mean (SD): 18.26 (2.08)                Mean (SD): 35.70 (3.04)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 14.00 /      `<br><br>`{=html} min/max 28.00 /
                                       22.00 `<br><br>`{=html} α raw=-0.01    24.00 `<br><br>`{=html} α raw=-0.13    43.00 `<br><br>`{=html} α raw=0.07
                                       `<br>`{=html} α std=-0.01              `<br>`{=html} α std=-0.15              `<br>`{=html} α std=0.08
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.05, p=0.000`<br><br>`{=html}Mean   t=8.40, p=0.000`<br><br>`{=html}Mean   t=7.89, p=0.000`<br><br>`{=html}Mean
                                       diff=2.56 `<br>`{=html} Cohen's        diff=3.57 `<br>`{=html} Cohen's        diff=6.13 `<br>`{=html} Cohen's
                                       d=0.49`<br><br>`{=html}Distribution    d=0.69`<br><br>`{=html}Distribution    d=0.62`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.55, JS=0.57, WD=0.79             TVD=0.54, JS=0.57, WD=0.77             TVD=0.54, JS=0.57, WD=0.78

     28 granite4_350m_t0.5.csv         Mean (SD): 17.51 (2.63)                Mean (SD): 17.95 (2.72)                Mean (SD): 35.46 (3.99)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 24.00 /
                                       24.00 `<br><br>`{=html} α raw=0.20     25.00 `<br><br>`{=html} α raw=0.14     45.00 `<br><br>`{=html} α raw=0.25
                                       `<br>`{=html} α std=0.20               `<br>`{=html} α std=0.15               `<br>`{=html} α std=0.25
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=5.73, p=0.000`<br><br>`{=html}Mean   t=7.12, p=0.000`<br><br>`{=html}Mean   t=7.19, p=0.000`<br><br>`{=html}Mean
                                       diff=2.62 `<br>`{=html} Cohen's        diff=3.27 `<br>`{=html} Cohen's        diff=5.88 `<br>`{=html} Cohen's
                                       d=0.49`<br><br>`{=html}Distribution    d=0.62`<br><br>`{=html}Distribution    d=0.59`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.45, JS=0.44, WD=0.61             TVD=0.41, JS=0.42, WD=0.56             TVD=0.43, JS=0.43, WD=0.58

     29 granite4_350m_t0.75.csv        Mean (SD): 17.40 (2.95)                Mean (SD): 18.51 (2.85)                Mean (SD): 35.91 (4.17)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 26.00 /
                                       26.00 `<br><br>`{=html} α raw=0.08     26.00 `<br><br>`{=html} α raw=-0.01    45.00 `<br><br>`{=html} α raw=0.07
                                       `<br>`{=html} α std=0.06               `<br>`{=html} α std=-0.01              `<br>`{=html} α std=0.09
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.97`<br><br>`{=html}t-test vs human:  0.97`<br><br>`{=html}t-test vs human:  0.97`<br><br>`{=html}t-test vs human:
                                       t=5.25, p=0.000`<br><br>`{=html}Mean   t=8.17, p=0.000`<br><br>`{=html}Mean   t=7.64, p=0.000`<br><br>`{=html}Mean
                                       diff=2.51 `<br>`{=html} Cohen's        diff=3.83 `<br>`{=html} Cohen's        diff=6.33 `<br>`{=html} Cohen's
                                       d=0.47`<br><br>`{=html}Distribution    d=0.72`<br><br>`{=html}Distribution    d=0.63`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.41, JS=0.40, WD=0.56             TVD=0.37, JS=0.38, WD=0.64             TVD=0.38, JS=0.39, WD=0.58

     30 granite4_350m_t1.0.csv         Mean (SD): 17.83 (2.79)                Mean (SD): 18.48 (2.81)                Mean (SD): 36.31 (3.96)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 27.00 /
                                       24.00 `<br><br>`{=html} α raw=-0.09    25.00 `<br><br>`{=html} α raw=-0.04    45.00 `<br><br>`{=html} α raw=-0.06
                                       `<br>`{=html} α std=-0.11              `<br>`{=html} α std=-0.04              `<br>`{=html} α std=-0.07
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.91`<br><br>`{=html}t-test vs human:  0.91`<br><br>`{=html}t-test vs human:  0.91`<br><br>`{=html}t-test vs human:
                                       t=6.26, p=0.000`<br><br>`{=html}Mean   t=8.12, p=0.000`<br><br>`{=html}Mean   t=8.21, p=0.000`<br><br>`{=html}Mean
                                       diff=2.94 `<br>`{=html} Cohen's        diff=3.79 `<br>`{=html} Cohen's        diff=6.74 `<br>`{=html} Cohen's
                                       d=0.55`<br><br>`{=html}Distribution    d=0.72`<br><br>`{=html}Distribution    d=0.67`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.31, JS=0.31, WD=0.49             TVD=0.32, JS=0.32, WD=0.63             TVD=0.32, JS=0.31, WD=0.56

     31 granite4_350m_t1.2.csv         Mean (SD): 18.26 (2.63)                Mean (SD): 17.93 (3.40)                Mean (SD): 36.19 (4.24)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 26.00 /
                                       25.00 `<br><br>`{=html} α raw=-0.36    24.00 `<br><br>`{=html} α raw=0.15     47.00 `<br><br>`{=html} α raw=-0.07
                                       `<br>`{=html} α std=-0.36              `<br>`{=html} α std=0.15               `<br>`{=html} α std=-0.09
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.99`<br><br>`{=html}t-test vs human:  0.99`<br><br>`{=html}t-test vs human:  0.99`<br><br>`{=html}t-test vs human:
                                       t=7.32, p=0.000`<br><br>`{=html}Mean   t=6.40, p=0.000`<br><br>`{=html}Mean   t=7.92, p=0.000`<br><br>`{=html}Mean
                                       diff=3.37 `<br>`{=html} Cohen's        diff=3.24 `<br>`{=html} Cohen's        diff=6.61 `<br>`{=html} Cohen's
                                       d=0.63`<br><br>`{=html}Distribution    d=0.60`<br><br>`{=html}Distribution    d=0.66`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.29, JS=0.30, WD=0.56             TVD=0.30, JS=0.31, WD=0.54             TVD=0.30, JS=0.30, WD=0.55

     32 granite4_350m_t1.4.csv         Mean (SD): 17.56 (3.09)                Mean (SD): 18.02 (3.07)                Mean (SD): 35.58 (4.46)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 27.00 /
                                       25.00 `<br><br>`{=html} α raw=0.01     25.00 `<br><br>`{=html} α raw=0.08     44.00 `<br><br>`{=html} α raw=0.09
                                       `<br>`{=html} α std=0.01               `<br>`{=html} α std=0.06               `<br>`{=html} α std=0.09
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.99`<br><br>`{=html}t-test vs human:  0.99`<br><br>`{=html}t-test vs human:  0.99`<br><br>`{=html}t-test vs human:
                                       t=5.48, p=0.000`<br><br>`{=html}Mean   t=6.92, p=0.000`<br><br>`{=html}Mean   t=7.11, p=0.000`<br><br>`{=html}Mean
                                       diff=2.67 `<br>`{=html} Cohen's        diff=3.34 `<br>`{=html} Cohen's        diff=6.00 `<br>`{=html} Cohen's
                                       d=0.50`<br><br>`{=html}Distribution    d=0.63`<br><br>`{=html}Distribution    d=0.60`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.27, JS=0.28, WD=0.45             TVD=0.29, JS=0.30, WD=0.56             TVD=0.28, JS=0.29, WD=0.50

     33 granite4_350m_t1.6.csv         Mean (SD): 17.11 (2.83)                Mean (SD): 17.86 (3.20)                Mean (SD): 34.97 (3.84)
                                       `<br><br>`{=html} min/max 9.00 / 26.00 `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 25.00 /
                                       `<br><br>`{=html} α raw=-0.36          27.00 `<br><br>`{=html} α raw=-0.01    45.00 `<br><br>`{=html} α raw=-0.44
                                       `<br>`{=html} α std=-0.36              `<br>`{=html} α std=-0.00              `<br>`{=html} α std=-0.44
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.92`<br><br>`{=html}t-test vs human:  0.92`<br><br>`{=html}t-test vs human:  0.92`<br><br>`{=html}t-test vs human:
                                       t=4.67, p=0.000`<br><br>`{=html}Mean   t=6.42, p=0.000`<br><br>`{=html}Mean   t=6.60, p=0.000`<br><br>`{=html}Mean
                                       diff=2.21 `<br>`{=html} Cohen's        diff=3.18 `<br>`{=html} Cohen's        diff=5.39 `<br>`{=html} Cohen's
                                       d=0.41`<br><br>`{=html}Distribution    d=0.59`<br><br>`{=html}Distribution    d=0.54`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.28, JS=0.27, WD=0.41             TVD=0.28, JS=0.29, WD=0.53             TVD=0.27, JS=0.28, WD=0.45

     34 granite4_350m_t1.8.csv         Mean (SD): 17.87 (2.86)                Mean (SD): 18.00 (2.84)                Mean (SD): 35.87 (4.04)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 24.00 /
                                       26.00 `<br><br>`{=html} α raw=-0.22    24.00 `<br><br>`{=html} α raw=-0.33    46.00 `<br><br>`{=html} α raw=-0.24
                                       `<br>`{=html} α std=-0.22              `<br>`{=html} α std=-0.33              `<br>`{=html} α std=-0.24
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.28, p=0.000`<br><br>`{=html}Mean   t=7.05, p=0.000`<br><br>`{=html}Mean   t=7.63, p=0.000`<br><br>`{=html}Mean
                                       diff=2.98 `<br>`{=html} Cohen's        diff=3.32 `<br>`{=html} Cohen's        diff=6.30 `<br>`{=html} Cohen's
                                       d=0.56`<br><br>`{=html}Distribution    d=0.62`<br><br>`{=html}Distribution    d=0.63`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.24, JS=0.25, WD=0.50             TVD=0.29, JS=0.29, WD=0.55             TVD=0.25, JS=0.27, WD=0.52

     35 granite4_350m_t2.0.csv         Mean (SD): 17.81 (3.40)                Mean (SD): 18.11 (2.95)                Mean (SD): 35.93 (4.27)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 26.00 /
                                       25.00 `<br><br>`{=html} α raw=0.15     24.00 `<br><br>`{=html} α raw=-0.06    46.00 `<br><br>`{=html} α raw=-0.06
                                       `<br>`{=html} α std=0.14               `<br>`{=html} α std=-0.08              `<br>`{=html} α std=-0.07
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       0.99`<br><br>`{=html}t-test vs human:  0.99`<br><br>`{=html}t-test vs human:  0.99`<br><br>`{=html}t-test vs human:
                                       t=5.75, p=0.000`<br><br>`{=html}Mean   t=7.21, p=0.000`<br><br>`{=html}Mean   t=7.61, p=0.000`<br><br>`{=html}Mean
                                       diff=2.92 `<br>`{=html} Cohen's        diff=3.43 `<br>`{=html} Cohen's        diff=6.35 `<br>`{=html} Cohen's
                                       d=0.54`<br><br>`{=html}Distribution    d=0.65`<br><br>`{=html}Distribution    d=0.63`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.24, JS=0.25, WD=0.49             TVD=0.27, JS=0.26, WD=0.57             TVD=0.25, JS=0.25, WD=0.53

     36 granite4_3b_t0.0.csv           Mean (SD): 15.09 (1.31)                Mean (SD): 15.82 (1.38)                Mean (SD): 30.91 (2.44)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 23.00 /
                                       18.00 `<br><br>`{=html} α raw=0.35     18.00 `<br><br>`{=html} α raw=0.21     35.00 `<br><br>`{=html} α raw=0.58
                                       `<br>`{=html} α std=0.39               `<br>`{=html} α std=0.19               `<br>`{=html} α std=0.57
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=0.50, p=0.618`<br><br>`{=html}Mean   t=2.88, p=0.004`<br><br>`{=html}Mean   t=1.77, p=0.078`<br><br>`{=html}Mean
                                       diff=0.20 `<br>`{=html} Cohen's        diff=1.14 `<br>`{=html} Cohen's        diff=1.33 `<br>`{=html} Cohen's
                                       d=0.04`<br><br>`{=html}Distribution    d=0.22`<br><br>`{=html}Distribution    d=0.14`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.45, JS=0.51, WD=0.64             TVD=0.44, JS=0.51, WD=0.71             TVD=0.44, JS=0.51, WD=0.67

     37 granite4_3b_t0.25.csv          Mean (SD): 15.53 (1.37)                Mean (SD): 16.15 (1.57)                Mean (SD): 31.68 (2.60)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 23.00 /
                                       19.00 `<br><br>`{=html} α raw=0.13     19.00 `<br><br>`{=html} α raw=0.31     36.00 `<br><br>`{=html} α raw=0.53
                                       `<br>`{=html} α std=0.10               `<br>`{=html} α std=0.29               `<br>`{=html} α std=0.48
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=1.60, p=0.110`<br><br>`{=html}Mean   t=3.65, p=0.000`<br><br>`{=html}Mean   t=2.77, p=0.006`<br><br>`{=html}Mean
                                       diff=0.64 `<br>`{=html} Cohen's        diff=1.47 `<br>`{=html} Cohen's        diff=2.10 `<br>`{=html} Cohen's
                                       d=0.12`<br><br>`{=html}Distribution    d=0.29`<br><br>`{=html}Distribution    d=0.21`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.42, JS=0.48, WD=0.67             TVD=0.44, JS=0.51, WD=0.73             TVD=0.42, JS=0.49, WD=0.70

     38 granite4_3b_t0.5.csv           Mean (SD): 15.89 (1.75)                Mean (SD): 16.36 (1.73)                Mean (SD): 32.25 (3.02)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 24.00 /
                                       20.00 `<br><br>`{=html} α raw=0.16     21.00 `<br><br>`{=html} α raw=0.19     39.00 `<br><br>`{=html} α raw=0.47
                                       `<br>`{=html} α std=0.12               `<br>`{=html} α std=0.11               `<br>`{=html} α std=0.42
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=2.42, p=0.016`<br><br>`{=html}Mean   t=4.11, p=0.000`<br><br>`{=html}Mean   t=3.45, p=0.001`<br><br>`{=html}Mean
                                       diff=1.00 `<br>`{=html} Cohen's        diff=1.68 `<br>`{=html} Cohen's        diff=2.67 `<br>`{=html} Cohen's
                                       d=0.19`<br><br>`{=html}Distribution    d=0.33`<br><br>`{=html}Distribution    d=0.27`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.37, JS=0.42, WD=0.62             TVD=0.37, JS=0.45, WD=0.65             TVD=0.37, JS=0.43, WD=0.63

     39 granite4_3b_t0.75.csv          Mean (SD): 16.06 (1.72)                Mean (SD): 16.47 (1.89)                Mean (SD): 32.53 (2.86)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 23.00 /
                                       21.00 `<br><br>`{=html} α raw=0.09     21.00 `<br><br>`{=html} α raw=0.16     39.00 `<br><br>`{=html} α raw=0.32
                                       `<br>`{=html} α std=0.10               `<br>`{=html} α std=0.11               `<br>`{=html} α std=0.29
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=2.84, p=0.005`<br><br>`{=html}Mean   t=4.31, p=0.000`<br><br>`{=html}Mean   t=3.84, p=0.000`<br><br>`{=html}Mean
                                       diff=1.17 `<br>`{=html} Cohen's        diff=1.79 `<br>`{=html} Cohen's        diff=2.95 `<br>`{=html} Cohen's
                                       d=0.22`<br><br>`{=html}Distribution    d=0.35`<br><br>`{=html}Distribution    d=0.30`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.35, JS=0.40, WD=0.59             TVD=0.32, JS=0.41, WD=0.58             TVD=0.33, JS=0.40, WD=0.59

     40 granite4_3b_t1.0.csv           Mean (SD): 16.32 (1.83)                Mean (SD): 16.94 (1.84)                Mean (SD): 33.26 (2.88)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 26.00 /
                                       22.00 `<br><br>`{=html} α raw=-0.04    22.00 `<br><br>`{=html} α raw=0.09     42.00 `<br><br>`{=html} α raw=0.23
                                       `<br>`{=html} α std=0.09               `<br>`{=html} α std=0.07               `<br>`{=html} α std=0.28
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.43, p=0.001`<br><br>`{=html}Mean   t=5.47, p=0.000`<br><br>`{=html}Mean   t=4.79, p=0.000`<br><br>`{=html}Mean
                                       diff=1.43 `<br>`{=html} Cohen's        diff=2.26 `<br>`{=html} Cohen's        diff=3.68 `<br>`{=html} Cohen's
                                       d=0.27`<br><br>`{=html}Distribution    d=0.44`<br><br>`{=html}Distribution    d=0.37`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.31, JS=0.36, WD=0.55             TVD=0.34, JS=0.42, WD=0.62             TVD=0.33, JS=0.39, WD=0.58

     41 granite4_3b_t1.2.csv           Mean (SD): 16.05 (2.00)                Mean (SD): 16.89 (1.89)                Mean (SD): 32.94 (3.19)
                                       `<br><br>`{=html} min/max 9.00 / 21.00 `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 22.00 /
                                       `<br><br>`{=html} α raw=0.14           21.00 `<br><br>`{=html} α raw=0.00     41.00 `<br><br>`{=html} α raw=0.33
                                       `<br>`{=html} α std=0.13               `<br>`{=html} α std=-0.02              `<br>`{=html} α std=0.32
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=2.73, p=0.007`<br><br>`{=html}Mean   t=5.32, p=0.000`<br><br>`{=html}Mean   t=4.30, p=0.000`<br><br>`{=html}Mean
                                       diff=1.16 `<br>`{=html} Cohen's        diff=2.21 `<br>`{=html} Cohen's        diff=3.36 `<br>`{=html} Cohen's
                                       d=0.22`<br><br>`{=html}Distribution    d=0.43`<br><br>`{=html}Distribution    d=0.34`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.29, JS=0.35, WD=0.52             TVD=0.33, JS=0.40, WD=0.59             TVD=0.31, JS=0.37, WD=0.55

     42 granite4_3b_t1.4.csv           Mean (SD): 16.23 (2.22)                Mean (SD): 17.01 (1.87)                Mean (SD): 33.24 (3.33)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 25.00 /
                                       21.00 `<br><br>`{=html} α raw=0.25     21.00 `<br><br>`{=html} α raw=-0.10    39.00 `<br><br>`{=html} α raw=0.34
                                       `<br>`{=html} α std=0.24               `<br>`{=html} α std=-0.16              `<br>`{=html} α std=0.31
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.08, p=0.002`<br><br>`{=html}Mean   t=5.62, p=0.000`<br><br>`{=html}Mean   t=4.65, p=0.000`<br><br>`{=html}Mean
                                       diff=1.34 `<br>`{=html} Cohen's        diff=2.33 `<br>`{=html} Cohen's        diff=3.66 `<br>`{=html} Cohen's
                                       d=0.25`<br><br>`{=html}Distribution    d=0.45`<br><br>`{=html}Distribution    d=0.37`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.25, JS=0.33, WD=0.48             TVD=0.32, JS=0.39, WD=0.57             TVD=0.28, JS=0.35, WD=0.53

     43 granite4_3b_t1.6.csv           Mean (SD): 16.71 (2.10)                Mean (SD): 17.14 (1.94)                Mean (SD): 33.85 (3.39)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 26.00 /
                                       23.00 `<br><br>`{=html} α raw=0.15     21.00 `<br><br>`{=html} α raw=-0.09    44.00 `<br><br>`{=html} α raw=0.34
                                       `<br>`{=html} α std=0.17               `<br>`{=html} α std=-0.11              `<br>`{=html} α std=0.34
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=4.24, p=0.000`<br><br>`{=html}Mean   t=5.89, p=0.000`<br><br>`{=html}Mean   t=5.41, p=0.000`<br><br>`{=html}Mean
                                       diff=1.82 `<br>`{=html} Cohen's        diff=2.46 `<br>`{=html} Cohen's        diff=4.27 `<br>`{=html} Cohen's
                                       d=0.35`<br><br>`{=html}Distribution    d=0.48`<br><br>`{=html}Distribution    d=0.43`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.30, JS=0.35, WD=0.53             TVD=0.31, JS=0.36, WD=0.52             TVD=0.30, JS=0.35, WD=0.53

     44 granite4_3b_t1.8.csv           Mean (SD): 16.57 (2.28)                Mean (SD): 16.70 (2.02)                Mean (SD): 33.27 (3.43)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 25.00 /
                                       23.00 `<br><br>`{=html} α raw=0.28     22.00 `<br><br>`{=html} α raw=-0.08    41.00 `<br><br>`{=html} α raw=0.32
                                       `<br>`{=html} α std=0.29               `<br>`{=html} α std=-0.10              `<br>`{=html} α std=0.32
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.83, p=0.000`<br><br>`{=html}Mean   t=4.79, p=0.000`<br><br>`{=html}Mean   t=4.66, p=0.000`<br><br>`{=html}Mean
                                       diff=1.68 `<br>`{=html} Cohen's        diff=2.02 `<br>`{=html} Cohen's        diff=3.69 `<br>`{=html} Cohen's
                                       d=0.32`<br><br>`{=html}Distribution    d=0.39`<br><br>`{=html}Distribution    d=0.37`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.31, JS=0.35, WD=0.54             TVD=0.27, JS=0.34, WD=0.50             TVD=0.29, JS=0.34, WD=0.52

     45 granite4_3b_t2.0.csv           Mean (SD): 16.63 (2.16)                Mean (SD): 17.15 (2.00)                Mean (SD): 33.78 (3.38)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 23.00 /
                                       21.00 `<br><br>`{=html} α raw=0.18     21.00 `<br><br>`{=html} α raw=0.09     41.00 `<br><br>`{=html} α raw=0.36
                                       `<br>`{=html} α std=0.20               `<br>`{=html} α std=0.08               `<br>`{=html} α std=0.35
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=4.03, p=0.000`<br><br>`{=html}Mean   t=5.87, p=0.000`<br><br>`{=html}Mean   t=5.32, p=0.000`<br><br>`{=html}Mean
                                       diff=1.74 `<br>`{=html} Cohen's        diff=2.47 `<br>`{=html} Cohen's        diff=4.20 `<br>`{=html} Cohen's
                                       d=0.33`<br><br>`{=html}Distribution    d=0.48`<br><br>`{=html}Distribution    d=0.42`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.30, JS=0.34, WD=0.51             TVD=0.33, JS=0.40, WD=0.57             TVD=0.31, JS=0.37, WD=0.54

     46 lfm2_24b_t0.0.csv              Mean (SD): 16.51 (1.82)                Mean (SD): 15.23 (1.93)                Mean (SD): 31.74 (3.39)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 25.00 /
                                       22.00 `<br><br>`{=html} α raw=0.43     22.00 `<br><br>`{=html} α raw=0.49     43.00 `<br><br>`{=html} α raw=0.68
                                       `<br>`{=html} α std=0.47               `<br>`{=html} α std=0.58               `<br>`{=html} α std=0.72
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.89, p=0.000`<br><br>`{=html}Mean   t=1.31, p=0.191`<br><br>`{=html}Mean   t=2.74, p=0.007`<br><br>`{=html}Mean
                                       diff=1.62 `<br>`{=html} Cohen's        diff=0.55 `<br>`{=html} Cohen's        diff=2.16 `<br>`{=html} Cohen's
                                       d=0.31`<br><br>`{=html}Distribution    d=0.11`<br><br>`{=html}Distribution    d=0.22`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.23, JS=0.28, WD=0.39             TVD=0.16, JS=0.19, WD=0.21             TVD=0.19, JS=0.23, WD=0.29

     47 lfm2_24b_t0.25.csv             Mean (SD): 16.42 (1.65)                Mean (SD): 15.43 (1.92)                Mean (SD): 31.85 (3.22)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 23.00 /
                                       21.00 `<br><br>`{=html} α raw=0.47     22.00 `<br><br>`{=html} α raw=0.57     42.00 `<br><br>`{=html} α raw=0.71
                                       `<br>`{=html} α std=0.56               `<br>`{=html} α std=0.61               `<br>`{=html} α std=0.76
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.74, p=0.000`<br><br>`{=html}Mean   t=1.79, p=0.074`<br><br>`{=html}Mean   t=2.90, p=0.004`<br><br>`{=html}Mean
                                       diff=1.53 `<br>`{=html} Cohen's        diff=0.75 `<br>`{=html} Cohen's        diff=2.27 `<br>`{=html} Cohen's
                                       d=0.29`<br><br>`{=html}Distribution    d=0.14`<br><br>`{=html}Distribution    d=0.23`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.24, JS=0.30, WD=0.38             TVD=0.19, JS=0.24, WD=0.25             TVD=0.21, JS=0.26, WD=0.31

     48 lfm2_24b_t0.75.csv             Mean (SD): 16.70 (1.68)                Mean (SD): 15.44 (2.04)                Mean (SD): 32.14 (3.38)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 24.00 /
                                       22.00 `<br><br>`{=html} α raw=0.40     23.00 `<br><br>`{=html} α raw=0.54     45.00 `<br><br>`{=html} α raw=0.69
                                       `<br>`{=html} α std=0.46               `<br>`{=html} α std=0.57               `<br>`{=html} α std=0.72
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=4.41, p=0.000`<br><br>`{=html}Mean   t=1.79, p=0.074`<br><br>`{=html}Mean   t=3.25, p=0.001`<br><br>`{=html}Mean
                                       diff=1.81 `<br>`{=html} Cohen's        diff=0.76 `<br>`{=html} Cohen's        diff=2.56 `<br>`{=html} Cohen's
                                       d=0.35`<br><br>`{=html}Distribution    d=0.15`<br><br>`{=html}Distribution    d=0.26`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.23, JS=0.27, WD=0.41             TVD=0.18, JS=0.21, WD=0.24             TVD=0.20, JS=0.24, WD=0.32

     49 lfm2_24b_t1.0.csv              Mean (SD): 16.53 (1.72)                Mean (SD): 15.43 (2.12)                Mean (SD): 31.96 (3.37)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 25.00 /
                                       22.00 `<br><br>`{=html} α raw=0.26     24.00 `<br><br>`{=html} α raw=0.49     46.00 `<br><br>`{=html} α raw=0.61
                                       `<br>`{=html} α std=0.34               `<br>`{=html} α std=0.52               `<br>`{=html} α std=0.67
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.98, p=0.000`<br><br>`{=html}Mean   t=1.75, p=0.081`<br><br>`{=html}Mean   t=3.02, p=0.003`<br><br>`{=html}Mean
                                       diff=1.64 `<br>`{=html} Cohen's        diff=0.75 `<br>`{=html} Cohen's        diff=2.38 `<br>`{=html} Cohen's
                                       d=0.31`<br><br>`{=html}Distribution    d=0.14`<br><br>`{=html}Distribution    d=0.24`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.21, JS=0.25, WD=0.37             TVD=0.18, JS=0.21, WD=0.24             TVD=0.20, JS=0.22, WD=0.30

     50 lfm2_24b_t1.2.csv              Mean (SD): 16.48 (1.55)                Mean (SD): 15.24 (1.96)                Mean (SD): 31.72 (3.00)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 25.00 /
                                       22.00 `<br><br>`{=html} α raw=0.18     21.00 `<br><br>`{=html} α raw=0.48     42.00 `<br><br>`{=html} α raw=0.56
                                       `<br>`{=html} α std=0.29               `<br>`{=html} α std=0.51               `<br>`{=html} α std=0.62
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.93, p=0.000`<br><br>`{=html}Mean   t=1.33, p=0.185`<br><br>`{=html}Mean   t=2.77, p=0.006`<br><br>`{=html}Mean
                                       diff=1.59 `<br>`{=html} Cohen's        diff=0.56 `<br>`{=html} Cohen's        diff=2.14 `<br>`{=html} Cohen's
                                       d=0.31`<br><br>`{=html}Distribution    d=0.11`<br><br>`{=html}Distribution    d=0.22`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.19, JS=0.22, WD=0.35             TVD=0.17, JS=0.19, WD=0.23             TVD=0.18, JS=0.20, WD=0.27

     51 lfm2_24b_t1.4.csv              Mean (SD): 16.54 (1.56)                Mean (SD): 15.29 (2.02)                Mean (SD): 31.83 (3.15)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 25.00 /
                                       21.00 `<br><br>`{=html} α raw=0.08     22.00 `<br><br>`{=html} α raw=0.41     42.00 `<br><br>`{=html} α raw=0.55
                                       `<br>`{=html} α std=0.21               `<br>`{=html} α std=0.47               `<br>`{=html} α std=0.63
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=4.07, p=0.000`<br><br>`{=html}Mean   t=1.44, p=0.151`<br><br>`{=html}Mean   t=2.89, p=0.004`<br><br>`{=html}Mean
                                       diff=1.65 `<br>`{=html} Cohen's        diff=0.61 `<br>`{=html} Cohen's        diff=2.25 `<br>`{=html} Cohen's
                                       d=0.32`<br><br>`{=html}Distribution    d=0.12`<br><br>`{=html}Distribution    d=0.23`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.20, JS=0.23, WD=0.36             TVD=0.16, JS=0.18, WD=0.23             TVD=0.18, JS=0.20, WD=0.27

     52 lfm2_24b_t1.6.csv              Mean (SD): 16.58 (1.83)                Mean (SD): 15.62 (2.10)                Mean (SD): 32.20 (3.30)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 23.00 /
                                       23.00 `<br><br>`{=html} α raw=0.34     23.00 `<br><br>`{=html} α raw=0.52     45.00 `<br><br>`{=html} α raw=0.60
                                       `<br>`{=html} α std=0.41               `<br>`{=html} α std=0.57               `<br>`{=html} α std=0.66
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=4.06, p=0.000`<br><br>`{=html}Mean   t=2.20, p=0.028`<br><br>`{=html}Mean   t=3.34, p=0.001`<br><br>`{=html}Mean
                                       diff=1.69 `<br>`{=html} Cohen's        diff=0.94 `<br>`{=html} Cohen's        diff=2.62 `<br>`{=html} Cohen's
                                       d=0.32`<br><br>`{=html}Distribution    d=0.18`<br><br>`{=html}Distribution    d=0.26`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.21, JS=0.24, WD=0.38             TVD=0.16, JS=0.19, WD=0.25             TVD=0.18, JS=0.21, WD=0.32

     53 lfm2_24b_t1.8.csv              Mean (SD): 16.69 (1.91)                Mean (SD): 15.67 (2.42)                Mean (SD): 32.36 (3.92)
                                       `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 25.00 /
                                       24.00 `<br><br>`{=html} α raw=0.35     22.00 `<br><br>`{=html} α raw=0.57     46.00 `<br><br>`{=html} α raw=0.69
                                       `<br>`{=html} α std=0.41               `<br>`{=html} α std=0.62               `<br>`{=html} α std=0.73
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=4.29, p=0.000`<br><br>`{=html}Mean   t=2.23, p=0.026`<br><br>`{=html}Mean   t=3.42, p=0.001`<br><br>`{=html}Mean
                                       diff=1.80 `<br>`{=html} Cohen's        diff=0.99 `<br>`{=html} Cohen's        diff=2.78 `<br>`{=html} Cohen's
                                       d=0.34`<br><br>`{=html}Distribution    d=0.19`<br><br>`{=html}Distribution    d=0.28`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.20, JS=0.23, WD=0.38             TVD=0.17, JS=0.19, WD=0.25             TVD=0.19, JS=0.21, WD=0.32

     54 lfm2_24b_t2.0.csv              Mean (SD): 16.40 (1.77)                Mean (SD): 15.74 (2.27)                Mean (SD): 32.14 (3.46)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 26.00 /
                                       22.00 `<br><br>`{=html} α raw=0.20     25.00 `<br><br>`{=html} α raw=0.46     44.00 `<br><br>`{=html} α raw=0.56
                                       `<br>`{=html} α std=0.28               `<br>`{=html} α std=0.49               `<br>`{=html} α std=0.61
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.65, p=0.000`<br><br>`{=html}Mean   t=2.43, p=0.015`<br><br>`{=html}Mean   t=3.23, p=0.001`<br><br>`{=html}Mean
                                       diff=1.51 `<br>`{=html} Cohen's        diff=1.06 `<br>`{=html} Cohen's        diff=2.56 `<br>`{=html} Cohen's
                                       d=0.29`<br><br>`{=html}Distribution    d=0.20`<br><br>`{=html}Distribution    d=0.26`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.23, JS=0.26, WD=0.36             TVD=0.17, JS=0.20, WD=0.27             TVD=0.20, JS=0.22, WD=0.32

     55 llama3.2_1b_t0.0.csv           Mean (SD): 17.89 (0.71)                Mean (SD): 18.00 (0.00)                Mean (SD): 35.89 (0.71)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 18.00 /      `<br><br>`{=html} min/max 30.00 /
                                       18.00 `<br><br>`{=html} α raw=N/A      18.00 `<br><br>`{=html} α raw=N/A      36.00 `<br><br>`{=html} α raw=N/A
                                       (constant cols) `<br>`{=html} α        (constant cols) `<br>`{=html} α        (constant cols) `<br>`{=html} α
                                       std=N/A (constant cols                 std=N/A (constant cols                 std=N/A (constant cols
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.97, p=0.000`<br><br>`{=html}Mean   t=8.98, p=0.000`<br><br>`{=html}Mean   t=8.83, p=0.000`<br><br>`{=html}Mean
                                       diff=3.00 `<br>`{=html} Cohen's        diff=3.32 `<br>`{=html} Cohen's        diff=6.31 `<br>`{=html} Cohen's
                                       d=0.72`<br><br>`{=html}Distribution    d=0.82`<br><br>`{=html}Distribution    d=0.80`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.70, JS=0.71, WD=1.05             TVD=0.71, JS=0.73, WD=1.08             TVD=0.70, JS=0.72, WD=1.06

     56 llama3.2_1b_t0.25.csv          Mean (SD): 17.60 (2.18)                Mean (SD): 17.78 (1.55)                Mean (SD): 35.37 (2.85)
                                       `<br><br>`{=html} min/max 9.00 / 24.00 `<br><br>`{=html} min/max 9.00 / 24.00 `<br><br>`{=html} min/max 21.00 /
                                       `<br><br>`{=html} α raw=0.19           `<br><br>`{=html} α raw=0.05           45.00 `<br><br>`{=html} α raw=0.24
                                       `<br>`{=html} α std=0.22               `<br>`{=html} α std=0.02               `<br>`{=html} α std=0.24
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.86, p=0.000`<br><br>`{=html}Mean   t=8.14, p=0.000`<br><br>`{=html}Mean   t=7.91, p=0.000`<br><br>`{=html}Mean
                                       diff=2.70 `<br>`{=html} Cohen's        diff=3.09 `<br>`{=html} Cohen's        diff=5.80 `<br>`{=html} Cohen's
                                       d=0.61`<br><br>`{=html}Distribution    d=0.73`<br><br>`{=html}Distribution    d=0.71`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.60, JS=0.58, WD=0.92             TVD=0.65, JS=0.63, WD=1.00             TVD=0.62, JS=0.60, WD=0.96

     57 llama3.2_1b_t0.5.csv           Mean (SD): 17.49 (2.75)                Mean (SD): 17.67 (2.57)                Mean (SD): 35.16 (3.88)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 23.00 /
                                       26.00 `<br><br>`{=html} α raw=0.03     `<br><br>`{=html} α raw=0.03           46.00 `<br><br>`{=html} α raw=0.09
                                       `<br>`{=html} α std=0.01               `<br>`{=html} α std=0.02               `<br>`{=html} α std=0.06
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.40, p=0.000`<br><br>`{=html}Mean   t=7.49, p=0.000`<br><br>`{=html}Mean   t=7.46, p=0.000`<br><br>`{=html}Mean
                                       diff=2.60 `<br>`{=html} Cohen's        diff=2.98 `<br>`{=html} Cohen's        diff=5.58 `<br>`{=html} Cohen's
                                       d=0.57`<br><br>`{=html}Distribution    d=0.67`<br><br>`{=html}Distribution    d=0.67`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.43, JS=0.43, WD=0.76             TVD=0.46, JS=0.47, WD=0.82             TVD=0.44, JS=0.44, WD=0.79

     58 llama3.2_1b_t0.75.csv          Mean (SD): 17.75 (2.94)                Mean (SD): 17.67 (2.79)                Mean (SD): 35.43 (4.12)
                                       `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 25.00 /
                                       `<br><br>`{=html} α raw=-0.05          25.00 `<br><br>`{=html} α raw=-0.07    50.00 `<br><br>`{=html} α raw=-0.02
                                       `<br>`{=html} α std=-0.06              `<br>`{=html} α std=-0.07              `<br>`{=html} α std=-0.02
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.97, p=0.000`<br><br>`{=html}Mean   t=7.42, p=0.000`<br><br>`{=html}Mean   t=7.78, p=0.000`<br><br>`{=html}Mean
                                       diff=2.86 `<br>`{=html} Cohen's        diff=2.99 `<br>`{=html} Cohen's        diff=5.85 `<br>`{=html} Cohen's
                                       d=0.62`<br><br>`{=html}Distribution    d=0.66`<br><br>`{=html}Distribution    d=0.70`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.33, JS=0.34, WD=0.67             TVD=0.35, JS=0.38, WD=0.72             TVD=0.34, JS=0.36, WD=0.69

     59 llama3.2_1b_t1.0.csv           Mean (SD): 17.84 (3.07)                Mean (SD): 17.82 (2.91)                Mean (SD): 35.65 (4.32)
                                       `<br><br>`{=html} min/max 9.00 / 26.00 `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 24.00 /
                                       `<br><br>`{=html} α raw=-0.05          27.00 `<br><br>`{=html} α raw=-0.05    47.00 `<br><br>`{=html} α raw=0.00
                                       `<br>`{=html} α std=-0.04              `<br>`{=html} α std=-0.05              `<br>`{=html} α std=0.01
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.12, p=0.000`<br><br>`{=html}Mean   t=7.72, p=0.000`<br><br>`{=html}Mean   t=8.04, p=0.000`<br><br>`{=html}Mean
                                       diff=2.94 `<br>`{=html} Cohen's        diff=3.13 `<br>`{=html} Cohen's        diff=6.08 `<br>`{=html} Cohen's
                                       d=0.63`<br><br>`{=html}Distribution    d=0.68`<br><br>`{=html}Distribution    d=0.72`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.26, JS=0.28, WD=0.58             TVD=0.31, JS=0.34, WD=0.66             TVD=0.28, JS=0.31, WD=0.62

     60 llama3.2_1b_t1.2.csv           Mean (SD): 17.39 (3.10)                Mean (SD): 17.70 (3.34)                Mean (SD): 35.09 (4.49)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 24.00 /
                                       27.00 `<br><br>`{=html} α raw=0.05     25.00 `<br><br>`{=html} α raw=0.20     46.00 `<br><br>`{=html} α raw=0.09
                                       `<br>`{=html} α std=0.05               `<br>`{=html} α std=0.21               `<br>`{=html} α std=0.09
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.04, p=0.000`<br><br>`{=html}Mean   t=7.23, p=0.000`<br><br>`{=html}Mean   t=7.26, p=0.000`<br><br>`{=html}Mean
                                       diff=2.50 `<br>`{=html} Cohen's        diff=3.01 `<br>`{=html} Cohen's        diff=5.51 `<br>`{=html} Cohen's
                                       d=0.53`<br><br>`{=html}Distribution    d=0.64`<br><br>`{=html}Distribution    d=0.65`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.24, JS=0.24, WD=0.50             TVD=0.26, JS=0.29, WD=0.58             TVD=0.25, JS=0.27, WD=0.54

     61 llama3.2_1b_t1.4.csv           Mean (SD): 17.88 (3.20)                Mean (SD): 17.87 (3.04)                Mean (SD): 35.75 (4.59)
                                       `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 9.00 / 26.00 `<br><br>`{=html} min/max 23.00 /
                                       `<br><br>`{=html} α raw=0.05           `<br><br>`{=html} α raw=0.04           50.00 `<br><br>`{=html} α raw=0.12
                                       `<br>`{=html} α std=0.04               `<br>`{=html} α std=0.04               `<br>`{=html} α std=0.11
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.16, p=0.000`<br><br>`{=html}Mean   t=7.79, p=0.000`<br><br>`{=html}Mean   t=8.11, p=0.000`<br><br>`{=html}Mean
                                       diff=2.98 `<br>`{=html} Cohen's        diff=3.19 `<br>`{=html} Cohen's        diff=6.17 `<br>`{=html} Cohen's
                                       d=0.63`<br><br>`{=html}Distribution    d=0.69`<br><br>`{=html}Distribution    d=0.72`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.24, JS=0.24, WD=0.53             TVD=0.27, JS=0.29, WD=0.59             TVD=0.25, JS=0.26, WD=0.56

     62 llama3.2_1b_t1.5.csv           Mean (SD): 17.72 (3.10)                Mean (SD): 17.89 (3.11)                Mean (SD): 35.61 (4.34)
                                       `<br><br>`{=html} min/max 9.00 / 25.00 `<br><br>`{=html} min/max 9.00 / 26.00 `<br><br>`{=html} min/max 26.00 /
                                       `<br><br>`{=html} α raw=0.01           `<br><br>`{=html} α raw=0.06           48.00 `<br><br>`{=html} α raw=0.00
                                       `<br>`{=html} α std=0.01               `<br>`{=html} α std=0.06               `<br>`{=html} α std=-0.00
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.82, p=0.000`<br><br>`{=html}Mean   t=7.81, p=0.000`<br><br>`{=html}Mean   t=7.98, p=0.000`<br><br>`{=html}Mean
                                       diff=2.82 `<br>`{=html} Cohen's        diff=3.21 `<br>`{=html} Cohen's        diff=6.03 `<br>`{=html} Cohen's
                                       d=0.60`<br><br>`{=html}Distribution    d=0.69`<br><br>`{=html}Distribution    d=0.71`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.23, JS=0.23, WD=0.50             TVD=0.27, JS=0.30, WD=0.60             TVD=0.25, JS=0.26, WD=0.55

     63 llama3.2_1b_t1.6.csv           Mean (SD): 17.92 (3.12)                Mean (SD): 17.63 (2.88)                Mean (SD): 35.55 (4.27)
                                       `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 22.00 /
                                       `<br><br>`{=html} α raw=0.06           26.00 `<br><br>`{=html} α raw=-0.16    48.00 `<br><br>`{=html} α raw=-0.02
                                       `<br>`{=html} α std=0.06               `<br>`{=html} α std=-0.15              `<br>`{=html} α std=-0.02
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.30, p=0.000`<br><br>`{=html}Mean   t=7.27, p=0.000`<br><br>`{=html}Mean   t=7.91, p=0.000`<br><br>`{=html}Mean
                                       diff=3.03 `<br>`{=html} Cohen's        diff=2.94 `<br>`{=html} Cohen's        diff=5.97 `<br>`{=html} Cohen's
                                       d=0.64`<br><br>`{=html}Distribution    d=0.64`<br><br>`{=html}Distribution    d=0.71`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.23, WD=0.51             TVD=0.23, JS=0.27, WD=0.52             TVD=0.23, JS=0.24, WD=0.52

     64 llama3.2_1b_t1.7.csv           Mean (SD): 17.58 (3.15)                Mean (SD): 17.94 (3.11)                Mean (SD): 35.52 (4.26)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 24.00 /
                                       26.00 `<br><br>`{=html} α raw=0.06     26.00 `<br><br>`{=html} α raw=0.04     49.00 `<br><br>`{=html} α raw=-0.04
                                       `<br>`{=html} α std=0.06               `<br>`{=html} α std=0.04               `<br>`{=html} α std=-0.05
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.47, p=0.000`<br><br>`{=html}Mean   t=7.94, p=0.000`<br><br>`{=html}Mean   t=7.88, p=0.000`<br><br>`{=html}Mean
                                       diff=2.69 `<br>`{=html} Cohen's        diff=3.26 `<br>`{=html} Cohen's        diff=5.95 `<br>`{=html} Cohen's
                                       d=0.57`<br><br>`{=html}Distribution    d=0.70`<br><br>`{=html}Distribution    d=0.70`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.19, JS=0.21, WD=0.47             TVD=0.25, JS=0.28, WD=0.56             TVD=0.22, JS=0.24, WD=0.51

     65 llama3.2_1b_t1.8.csv           Mean (SD): 18.01 (2.77)                Mean (SD): 17.83 (3.29)                Mean (SD): 35.85 (4.35)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 9.00 / 26.00 `<br><br>`{=html} min/max 25.00 /
                                       25.00 `<br><br>`{=html} α raw=-0.25    `<br><br>`{=html} α raw=0.19           50.00 `<br><br>`{=html} α raw=0.03
                                       `<br>`{=html} α std=-0.26              `<br>`{=html} α std=0.19               `<br>`{=html} α std=0.02
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.68, p=0.000`<br><br>`{=html}Mean   t=7.58, p=0.000`<br><br>`{=html}Mean   t=8.29, p=0.000`<br><br>`{=html}Mean
                                       diff=3.12 `<br>`{=html} Cohen's        diff=3.15 `<br>`{=html} Cohen's        diff=6.27 `<br>`{=html} Cohen's
                                       d=0.68`<br><br>`{=html}Distribution    d=0.67`<br><br>`{=html}Distribution    d=0.74`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.23, WD=0.52             TVD=0.24, JS=0.26, WD=0.54             TVD=0.23, JS=0.24, WD=0.53

     66 llama3.2_1b_t1.9.csv           Mean (SD): 17.97 (3.24)                Mean (SD): 17.87 (3.14)                Mean (SD): 35.84 (4.59)
                                       `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 9.00 / 25.00 `<br><br>`{=html} min/max 24.00 /
                                       `<br><br>`{=html} α raw=0.15           `<br><br>`{=html} α raw=0.06           51.00 `<br><br>`{=html} α raw=0.13
                                       `<br>`{=html} α std=0.15               `<br>`{=html} α std=0.06               `<br>`{=html} α std=0.14
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.37, p=0.000`<br><br>`{=html}Mean   t=7.74, p=0.000`<br><br>`{=html}Mean   t=8.23, p=0.000`<br><br>`{=html}Mean
                                       diff=3.08 `<br>`{=html} Cohen's        diff=3.19 `<br>`{=html} Cohen's        diff=6.27 `<br>`{=html} Cohen's
                                       d=0.65`<br><br>`{=html}Distribution    d=0.68`<br><br>`{=html}Distribution    d=0.73`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.22, WD=0.51             TVD=0.24, JS=0.27, WD=0.55             TVD=0.23, JS=0.24, WD=0.53

     67 llama3.2_1b_t2.0.csv           Mean (SD): 17.68 (3.09)                Mean (SD): 17.49 (2.78)                Mean (SD): 35.17 (4.08)
                                       `<br><br>`{=html} min/max 9.00 / 25.00 `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 21.00 /
                                       `<br><br>`{=html} α raw=0.05           25.00 `<br><br>`{=html} α raw=-0.22    45.00 `<br><br>`{=html} α raw=-0.11
                                       `<br>`{=html} α std=0.04               `<br>`{=html} α std=-0.22              `<br>`{=html} α std=-0.11
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.73, p=0.000`<br><br>`{=html}Mean   t=6.98, p=0.000`<br><br>`{=html}Mean   t=7.45, p=0.000`<br><br>`{=html}Mean
                                       diff=2.79 `<br>`{=html} Cohen's        diff=2.81 `<br>`{=html} Cohen's        diff=5.60 `<br>`{=html} Cohen's
                                       d=0.59`<br><br>`{=html}Distribution    d=0.62`<br><br>`{=html}Distribution    d=0.67`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.19, JS=0.20, WD=0.46             TVD=0.20, JS=0.23, WD=0.47             TVD=0.20, JS=0.21, WD=0.47

     68 nemotron-mini_4b_t0.0.csv      Mean (SD): 13.25 (0.86)                Mean (SD): 13.50 (0.77)                Mean (SD): 26.75 (1.32)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 24.00 /
                                       15.00 `<br><br>`{=html} α raw=N/A      16.00 `<br><br>`{=html} α raw=N/A      30.00 `<br><br>`{=html} α raw=N/A
                                       (constant cols) `<br>`{=html} α        (constant cols) `<br>`{=html} α        (constant cols) `<br>`{=html} α
                                       std=N/A (constant cols                 std=N/A (constant cols                 std=N/A (constant cols
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=-4.28, p=0.000`<br><br>`{=html}Mean  t=-3.14, p=0.002`<br><br>`{=html}Mean  t=-3.90, p=0.000`<br><br>`{=html}Mean
                                       diff=-1.64 `<br>`{=html} Cohen's       diff=-1.18 `<br>`{=html} Cohen's       diff=-2.83 `<br>`{=html} Cohen's
                                       d=-0.32`<br><br>`{=html}Distribution   d=-0.23`<br><br>`{=html}Distribution   d=-0.29`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.45, JS=0.49, WD=0.66             TVD=0.49, JS=0.55, WD=0.67             TVD=0.47, JS=0.51, WD=0.67

     69 nemotron-mini_4b_t0.25.csv     Mean (SD): 13.91 (1.69)                Mean (SD): 13.65 (1.18)                Mean (SD): 27.56 (2.22)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 22.00 /
                                       19.00 `<br><br>`{=html} α raw=N/A      16.00 `<br><br>`{=html} α raw=0.03     32.00 `<br><br>`{=html} α raw=N/A
                                       (constant cols) `<br>`{=html} α        `<br>`{=html} α std=0.03               (constant cols) `<br>`{=html} α
                                       std=N/A (constant cols                 `<br><br>`{=html}Prop Validated:       std=N/A (constant cols
                                       `<br><br>`{=html}Prop Validated:       1.00`<br><br>`{=html}t-test vs human:  `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  t=-2.67, p=0.008`<br><br>`{=html}Mean  1.00`<br><br>`{=html}t-test vs human:
                                       t=-2.39, p=0.017`<br><br>`{=html}Mean  diff=-1.03 `<br>`{=html} Cohen's       t=-2.70, p=0.007`<br><br>`{=html}Mean
                                       diff=-0.98 `<br>`{=html} Cohen's       d=-0.20`<br><br>`{=html}Distribution   diff=-2.02 `<br>`{=html} Cohen's
                                       d=-0.19`<br><br>`{=html}Distribution   distance vs human: `<br>`{=html}       d=-0.20`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       TVD=0.44, JS=0.46, WD=0.58             distance vs human: `<br>`{=html}
                                       TVD=0.40, JS=0.40, WD=0.54                                                    TVD=0.42, JS=0.42, WD=0.56

     70 nemotron-mini_4b_t0.5.csv      Mean (SD): 14.05 (1.78)                Mean (SD): 14.16 (1.82)                Mean (SD): 28.21 (2.33)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 22.00 /
                                       19.00 `<br><br>`{=html} α raw=-0.00    20.00 `<br><br>`{=html} α raw=-0.01    34.00 `<br><br>`{=html} α raw=-0.22
                                       `<br>`{=html} α std=0.06               `<br>`{=html} α std=0.01               `<br>`{=html} α std=-0.22
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=-2.04, p=0.043`<br><br>`{=html}Mean  t=-1.27, p=0.204`<br><br>`{=html}Mean  t=-1.82, p=0.070`<br><br>`{=html}Mean
                                       diff=-0.84 `<br>`{=html} Cohen's       diff=-0.52 `<br>`{=html} Cohen's       diff=-1.37 `<br>`{=html} Cohen's
                                       d=-0.16`<br><br>`{=html}Distribution   d=-0.10`<br><br>`{=html}Distribution   d=-0.14`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.34, JS=0.33, WD=0.43             TVD=0.30, JS=0.29, WD=0.34             TVD=0.32, JS=0.31, WD=0.39

     71 nemotron-mini_4b_t0.75.csv     Mean (SD): 14.60 (2.55)                Mean (SD): 14.20 (2.07)                Mean (SD): 28.80 (3.41)
                                       `<br><br>`{=html} min/max 9.00 / 21.00 `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 22.00 /
                                       `<br><br>`{=html} α raw=0.23           20.00 `<br><br>`{=html} α raw=-0.11    37.00 `<br><br>`{=html} α raw=0.16
                                       `<br>`{=html} α std=0.18               `<br>`{=html} α std=-0.10              `<br>`{=html} α std=0.10
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=-0.65, p=0.518`<br><br>`{=html}Mean  t=-1.15, p=0.253`<br><br>`{=html}Mean  t=-0.98, p=0.327`<br><br>`{=html}Mean
                                       diff=-0.29 `<br>`{=html} Cohen's       diff=-0.48 `<br>`{=html} Cohen's       diff=-0.78 `<br>`{=html} Cohen's
                                       d=-0.06`<br><br>`{=html}Distribution   d=-0.09`<br><br>`{=html}Distribution   d=-0.08`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.21, WD=0.25             TVD=0.21, JS=0.20, WD=0.25             TVD=0.22, JS=0.20, WD=0.25

     72 nemotron-mini_4b_t1.0.csv      Mean (SD): 14.78 (2.41)                Mean (SD): 14.29 (2.04)                Mean (SD): 29.07 (3.31)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 9.00 / 20.00 `<br><br>`{=html} min/max 20.00 /
                                       24.00 `<br><br>`{=html} α raw=0.03     `<br><br>`{=html} α raw=-0.35          37.00 `<br><br>`{=html} α raw=-0.01
                                       `<br>`{=html} α std=-0.04              `<br>`{=html} α std=-0.33              `<br>`{=html} α std=-0.03
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=-0.25, p=0.801`<br><br>`{=html}Mean  t=-0.94, p=0.350`<br><br>`{=html}Mean  t=-0.64, p=0.520`<br><br>`{=html}Mean
                                       diff=-0.11 `<br>`{=html} Cohen's       diff=-0.39 `<br>`{=html} Cohen's       diff=-0.51 `<br>`{=html} Cohen's
                                       d=-0.02`<br><br>`{=html}Distribution   d=-0.08`<br><br>`{=html}Distribution   d=-0.05`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.18, JS=0.21, WD=0.24             TVD=0.16, JS=0.18, WD=0.21             TVD=0.17, JS=0.19, WD=0.22

     73 nemotron-mini_4b_t1.2.csv      Mean (SD): 15.57 (2.43)                Mean (SD): 15.37 (2.50)                Mean (SD): 30.94 (3.60)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 8.00 / 21.00 `<br><br>`{=html} min/max 20.00 /
                                       22.00 `<br><br>`{=html} α raw=0.03     `<br><br>`{=html} α raw=0.07           39.00 `<br><br>`{=html} α raw=0.11
                                       `<br>`{=html} α std=0.00               `<br>`{=html} α std=0.09               `<br>`{=html} α std=0.12
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=1.52, p=0.129`<br><br>`{=html}Mean   t=1.54, p=0.125`<br><br>`{=html}Mean   t=1.71, p=0.089`<br><br>`{=html}Mean
                                       diff=0.68 `<br>`{=html} Cohen's        diff=0.69 `<br>`{=html} Cohen's        diff=1.36 `<br>`{=html} Cohen's
                                       d=0.13`<br><br>`{=html}Distribution    d=0.13`<br><br>`{=html}Distribution    d=0.14`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.18, JS=0.21, WD=0.28             TVD=0.15, JS=0.17, WD=0.25             TVD=0.17, JS=0.19, WD=0.26

     74 nemotron-mini_4b_t1.4.csv      Mean (SD): 15.33 (2.32)                Mean (SD): 14.60 (2.42)                Mean (SD): 29.93 (3.38)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 9.00 / 21.00 `<br><br>`{=html} min/max 21.00 /
                                       21.00 `<br><br>`{=html} α raw=-0.16    `<br><br>`{=html} α raw=-0.08          38.00 `<br><br>`{=html} α raw=-0.09
                                       `<br>`{=html} α std=-0.08              `<br>`{=html} α std=-0.08              `<br>`{=html} α std=-0.06
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=0.99, p=0.321`<br><br>`{=html}Mean   t=-0.19, p=0.848`<br><br>`{=html}Mean  t=0.45, p=0.655`<br><br>`{=html}Mean
                                       diff=0.44 `<br>`{=html} Cohen's        diff=-0.08 `<br>`{=html} Cohen's       diff=0.35 `<br>`{=html} Cohen's
                                       d=0.08`<br><br>`{=html}Distribution    d=-0.02`<br><br>`{=html}Distribution   d=0.04`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.17, JS=0.19, WD=0.23             TVD=0.15, JS=0.17, WD=0.20             TVD=0.15, JS=0.18, WD=0.21

     75 nemotron-mini_4b_t1.6.csv      Mean (SD): 15.38 (2.51)                Mean (SD): 14.98 (2.26)                Mean (SD): 30.36 (3.72)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 23.00 /
                                       22.00 `<br><br>`{=html} α raw=-0.05    23.00 `<br><br>`{=html} α raw=-0.20    45.00 `<br><br>`{=html} α raw=0.10
                                       `<br>`{=html} α std=-0.04              `<br>`{=html} α std=-0.19              `<br>`{=html} α std=0.15
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=1.08, p=0.280`<br><br>`{=html}Mean   t=0.68, p=0.495`<br><br>`{=html}Mean   t=0.97, p=0.331`<br><br>`{=html}Mean
                                       diff=0.49 `<br>`{=html} Cohen's        diff=0.30 `<br>`{=html} Cohen's        diff=0.78 `<br>`{=html} Cohen's
                                       d=0.09`<br><br>`{=html}Distribution    d=0.06`<br><br>`{=html}Distribution    d=0.08`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.18, JS=0.21, WD=0.25             TVD=0.16, JS=0.19, WD=0.23             TVD=0.17, JS=0.19, WD=0.24

     76 nemotron-mini_4b_t1.8.csv      Mean (SD): 15.04 (2.33)                Mean (SD): 15.18 (2.40)                Mean (SD): 30.22 (3.67)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 22.00 /
                                       21.00 `<br><br>`{=html} α raw=-0.23    22.00 `<br><br>`{=html} α raw=-0.10    41.00 `<br><br>`{=html} α raw=0.06
                                       `<br>`{=html} α std=-0.38              `<br>`{=html} α std=-0.09              `<br>`{=html} α std=0.04
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=0.34, p=0.737`<br><br>`{=html}Mean   t=1.13, p=0.261`<br><br>`{=html}Mean   t=0.80, p=0.423`<br><br>`{=html}Mean
                                       diff=0.15 `<br>`{=html} Cohen's        diff=0.50 `<br>`{=html} Cohen's        diff=0.64 `<br>`{=html} Cohen's
                                       d=0.03`<br><br>`{=html}Distribution    d=0.09`<br><br>`{=html}Distribution    d=0.06`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.16, JS=0.18, WD=0.21             TVD=0.16, JS=0.17, WD=0.21             TVD=0.15, JS=0.17, WD=0.21

     77 nemotron-mini_4b_t2.0.csv      Mean (SD): 15.01 (2.50)                Mean (SD): 14.77 (2.55)                Mean (SD): 29.78 (3.68)
                                       `<br><br>`{=html} min/max 9.00 / 24.00 `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 21.00 /
                                       `<br><br>`{=html} α raw=-0.01          24.00 `<br><br>`{=html} α raw=-0.01    39.00 `<br><br>`{=html} α raw=0.05
                                       `<br>`{=html} α std=-0.01              `<br>`{=html} α std=0.01               `<br>`{=html} α std=0.01
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=0.26, p=0.794`<br><br>`{=html}Mean   t=0.19, p=0.849`<br><br>`{=html}Mean   t=0.25, p=0.800`<br><br>`{=html}Mean
                                       diff=0.12 `<br>`{=html} Cohen's        diff=0.09 `<br>`{=html} Cohen's        diff=0.20 `<br>`{=html} Cohen's
                                       d=0.02`<br><br>`{=html}Distribution    d=0.02`<br><br>`{=html}Distribution    d=0.02`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.15, JS=0.18, WD=0.20             TVD=0.11, JS=0.13, WD=0.14             TVD=0.13, JS=0.15, WD=0.17

     78 olmo2_7b_t0.0.csv              Mean (SD): 17.23 (2.61)                Mean (SD): 16.04 (3.49)                Mean (SD): 33.27 (5.85)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 7.00 / 27.00 `<br><br>`{=html} min/max 19.00 /
                                       25.00 `<br><br>`{=html} α raw=0.65     `<br><br>`{=html} α raw=0.77           52.00 `<br><br>`{=html} α raw=0.85
                                       `<br>`{=html} α std=0.70               `<br>`{=html} α std=0.80               `<br>`{=html} α std=0.87
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=5.13, p=0.000`<br><br>`{=html}Mean   t=2.67, p=0.008`<br><br>`{=html}Mean   t=4.00, p=0.000`<br><br>`{=html}Mean
                                       diff=2.34 `<br>`{=html} Cohen's        diff=1.36 `<br>`{=html} Cohen's        diff=3.69 `<br>`{=html} Cohen's
                                       d=0.44`<br><br>`{=html}Distribution    d=0.25`<br><br>`{=html}Distribution    d=0.36`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.27, JS=0.31, WD=0.51             TVD=0.22, JS=0.26, WD=0.40             TVD=0.24, JS=0.28, WD=0.41

     79 olmo2_7b_t0.25.csv             Mean (SD): 17.57 (2.78)                Mean (SD): 17.00 (3.51)                Mean (SD): 34.57 (5.85)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 23.00 /
                                       27.00 `<br><br>`{=html} α raw=0.59     `<br><br>`{=html} α raw=0.71           54.00 `<br><br>`{=html} α raw=0.80
                                       `<br>`{=html} α std=0.63               `<br>`{=html} α std=0.72               `<br>`{=html} α std=0.81
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=5.75, p=0.000`<br><br>`{=html}Mean   t=4.54, p=0.000`<br><br>`{=html}Mean   t=5.41, p=0.000`<br><br>`{=html}Mean
                                       diff=2.68 `<br>`{=html} Cohen's        diff=2.32 `<br>`{=html} Cohen's        diff=4.99 `<br>`{=html} Cohen's
                                       d=0.50`<br><br>`{=html}Distribution    d=0.43`<br><br>`{=html}Distribution    d=0.49`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.28, JS=0.30, WD=0.52             TVD=0.25, JS=0.29, WD=0.42             TVD=0.26, JS=0.29, WD=0.46

     80 olmo2_7b_t0.5.csv              Mean (SD): 15.92 (2.94)                Mean (SD): 15.57 (3.57)                Mean (SD): 31.49 (5.74)
                                       `<br><br>`{=html} min/max 8.00 / 24.00 `<br><br>`{=html} min/max 8.00 / 25.00 `<br><br>`{=html} min/max 20.00 /
                                       `<br><br>`{=html} α raw=0.49           `<br><br>`{=html} α raw=0.68           43.00 `<br><br>`{=html} α raw=0.74
                                       `<br>`{=html} α std=0.54               `<br>`{=html} α std=0.70               `<br>`{=html} α std=0.77
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=2.16, p=0.031`<br><br>`{=html}Mean   t=1.73, p=0.086`<br><br>`{=html}Mean   t=2.09, p=0.038`<br><br>`{=html}Mean
                                       diff=1.03 `<br>`{=html} Cohen's        diff=0.89 `<br>`{=html} Cohen's        diff=1.91 `<br>`{=html} Cohen's
                                       d=0.19`<br><br>`{=html}Distribution    d=0.16`<br><br>`{=html}Distribution    d=0.19`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.17, JS=0.21, WD=0.30             TVD=0.20, JS=0.22, WD=0.33             TVD=0.17, JS=0.21, WD=0.32

     81 olmo2_7b_t0.75.csv             Mean (SD): 16.26 (3.26)                Mean (SD): 16.20 (3.22)                Mean (SD): 32.46 (5.59)
                                       `<br><br>`{=html} min/max 8.00 / 25.00 `<br><br>`{=html} min/max 9.00 / 25.00 `<br><br>`{=html} min/max 19.00 /
                                       `<br><br>`{=html} α raw=0.57           `<br><br>`{=html} α raw=0.50           47.00 `<br><br>`{=html} α raw=0.68
                                       `<br>`{=html} α std=0.60               `<br>`{=html} α std=0.50               `<br>`{=html} α std=0.71
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=2.76, p=0.006`<br><br>`{=html}Mean   t=3.09, p=0.002`<br><br>`{=html}Mean   t=3.18, p=0.002`<br><br>`{=html}Mean
                                       diff=1.37 `<br>`{=html} Cohen's        diff=1.52 `<br>`{=html} Cohen's        diff=2.88 `<br>`{=html} Cohen's
                                       d=0.25`<br><br>`{=html}Distribution    d=0.28`<br><br>`{=html}Distribution    d=0.28`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.21, JS=0.23, WD=0.35             TVD=0.22, JS=0.23, WD=0.33             TVD=0.21, JS=0.23, WD=0.34

     82 olmo2_7b_t1.0.csv              Mean (SD): 16.40 (2.83)                Mean (SD): 16.47 (3.20)                Mean (SD): 32.87 (5.16)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 8.00 / 24.00 `<br><br>`{=html} min/max 18.00 /
                                       25.00 `<br><br>`{=html} α raw=0.38     `<br><br>`{=html} α raw=0.47           46.00 `<br><br>`{=html} α raw=0.62
                                       `<br>`{=html} α std=0.45               `<br>`{=html} α std=0.48               `<br>`{=html} α std=0.64
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.22, p=0.001`<br><br>`{=html}Mean   t=3.66, p=0.000`<br><br>`{=html}Mean   t=3.74, p=0.000`<br><br>`{=html}Mean
                                       diff=1.51 `<br>`{=html} Cohen's        diff=1.79 `<br>`{=html} Cohen's        diff=3.29 `<br>`{=html} Cohen's
                                       d=0.28`<br><br>`{=html}Distribution    d=0.33`<br><br>`{=html}Distribution    d=0.32`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.25, JS=0.26, WD=0.37             TVD=0.22, JS=0.24, WD=0.34             TVD=0.23, JS=0.24, WD=0.35

     83 olmo2_7b_t1.2.csv              Mean (SD): 15.56 (3.29)                Mean (SD): 15.75 (3.56)                Mean (SD): 31.31 (5.93)
                                       `<br><br>`{=html} min/max 8.00 / 23.00 `<br><br>`{=html} min/max 7.00 / 24.00 `<br><br>`{=html} min/max 20.00 /
                                       `<br><br>`{=html} α raw=0.52           `<br><br>`{=html} α raw=0.51           46.00 `<br><br>`{=html} α raw=0.67
                                       `<br>`{=html} α std=0.55               `<br>`{=html} α std=0.52               `<br>`{=html} α std=0.70
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=1.34, p=0.181`<br><br>`{=html}Mean   t=2.08, p=0.039`<br><br>`{=html}Mean   t=1.87, p=0.063`<br><br>`{=html}Mean
                                       diff=0.67 `<br>`{=html} Cohen's        diff=1.07 `<br>`{=html} Cohen's        diff=1.73 `<br>`{=html} Cohen's
                                       d=0.12`<br><br>`{=html}Distribution    d=0.20`<br><br>`{=html}Distribution    d=0.17`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.16, JS=0.19, WD=0.28             TVD=0.15, JS=0.16, WD=0.20             TVD=0.15, JS=0.17, WD=0.24

     84 olmo2_7b_t1.4.csv              Mean (SD): 16.26 (2.90)                Mean (SD): 15.82 (3.35)                Mean (SD): 32.08 (5.38)
                                       `<br><br>`{=html} min/max 8.00 / 22.00 `<br><br>`{=html} min/max 9.00 / 23.00 `<br><br>`{=html} min/max 20.00 /
                                       `<br><br>`{=html} α raw=0.25           `<br><br>`{=html} α raw=0.47           45.00 `<br><br>`{=html} α raw=0.58
                                       `<br>`{=html} α std=0.30               `<br>`{=html} α std=0.49               `<br>`{=html} α std=0.61
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=2.89, p=0.004`<br><br>`{=html}Mean   t=2.28, p=0.023`<br><br>`{=html}Mean   t=2.80, p=0.005`<br><br>`{=html}Mean
                                       diff=1.37 `<br>`{=html} Cohen's        diff=1.14 `<br>`{=html} Cohen's        diff=2.50 `<br>`{=html} Cohen's
                                       d=0.26`<br><br>`{=html}Distribution    d=0.21`<br><br>`{=html}Distribution    d=0.25`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.20, JS=0.20, WD=0.28             TVD=0.19, JS=0.20, WD=0.27             TVD=0.19, JS=0.20, WD=0.28

     85 olmo2_7b_t1.6.csv              Mean (SD): 16.09 (3.73)                Mean (SD): 16.13 (3.04)                Mean (SD): 32.22 (5.99)
                                       `<br><br>`{=html} min/max 9.00 / 28.00 `<br><br>`{=html} min/max 9.00 / 23.00 `<br><br>`{=html} min/max 20.00 /
                                       `<br><br>`{=html} α raw=0.58           `<br><br>`{=html} α raw=0.24           51.00 `<br><br>`{=html} α raw=0.65
                                       `<br>`{=html} α std=0.60               `<br>`{=html} α std=0.26               `<br>`{=html} α std=0.67
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=2.27, p=0.024`<br><br>`{=html}Mean   t=3.02, p=0.003`<br><br>`{=html}Mean   t=2.84, p=0.005`<br><br>`{=html}Mean
                                       diff=1.20 `<br>`{=html} Cohen's        diff=1.45 `<br>`{=html} Cohen's        diff=2.64 `<br>`{=html} Cohen's
                                       d=0.22`<br><br>`{=html}Distribution    d=0.27`<br><br>`{=html}Distribution    d=0.26`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.17, JS=0.18, WD=0.26             TVD=0.18, JS=0.20, WD=0.24             TVD=0.17, JS=0.19, WD=0.25

     86 olmo2_7b_t1.8.csv              Mean (SD): 16.66 (3.04)                Mean (SD): 16.23 (3.69)                Mean (SD): 32.89 (5.81)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 8.00 / 25.00 `<br><br>`{=html} min/max 19.00 /
                                       24.00 `<br><br>`{=html} α raw=0.38     `<br><br>`{=html} α raw=0.51           47.00 `<br><br>`{=html} α raw=0.64
                                       `<br>`{=html} α std=0.41               `<br>`{=html} α std=0.51               `<br>`{=html} α std=0.65
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.67, p=0.000`<br><br>`{=html}Mean   t=2.96, p=0.003`<br><br>`{=html}Mean   t=3.60, p=0.000`<br><br>`{=html}Mean
                                       diff=1.77 `<br>`{=html} Cohen's        diff=1.55 `<br>`{=html} Cohen's        diff=3.31 `<br>`{=html} Cohen's
                                       d=0.33`<br><br>`{=html}Distribution    d=0.29`<br><br>`{=html}Distribution    d=0.32`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.23, JS=0.22, WD=0.37             TVD=0.17, JS=0.17, WD=0.26             TVD=0.20, JS=0.19, WD=0.31

     87 olmo2_7b_t2.0.csv              Mean (SD): 15.38 (3.40)                Mean (SD): 15.41 (3.59)                Mean (SD): 30.79 (5.98)
                                       `<br><br>`{=html} min/max 8.00 / 27.00 `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 21.00 /
                                       `<br><br>`{=html} α raw=0.49           25.00 `<br><br>`{=html} α raw=0.52     48.00 `<br><br>`{=html} α raw=0.66
                                       `<br>`{=html} α std=0.52               `<br>`{=html} α std=0.52               `<br>`{=html} α std=0.68
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=0.97, p=0.335`<br><br>`{=html}Mean   t=1.41, p=0.160`<br><br>`{=html}Mean   t=1.30, p=0.193`<br><br>`{=html}Mean
                                       diff=0.49 `<br>`{=html} Cohen's        diff=0.73 `<br>`{=html} Cohen's        diff=1.21 `<br>`{=html} Cohen's
                                       d=0.09`<br><br>`{=html}Distribution    d=0.13`<br><br>`{=html}Distribution    d=0.12`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.13, JS=0.15, WD=0.20             TVD=0.15, JS=0.17, WD=0.20             TVD=0.14, JS=0.15, WD=0.20

     88 qwen2.5-coder_0.5b_t0.0.csv    Mean (SD): 18.00 (0.06)                Mean (SD): 18.00 (0.00)                Mean (SD): 36.00 (0.06)
                                       `<br><br>`{=html} min/max 17.00 /      `<br><br>`{=html} min/max 18.00 /      `<br><br>`{=html} min/max 35.00 /
                                       18.00 `<br><br>`{=html} α raw=N/A      18.00 `<br><br>`{=html} α raw=N/A      36.00 `<br><br>`{=html} α raw=N/A
                                       (constant cols) `<br>`{=html} α        (constant cols) `<br>`{=html} α        (constant cols) `<br>`{=html} α
                                       std=N/A (constant cols                 std=N/A (constant cols                 std=N/A (constant cols
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=8.31, p=0.000`<br><br>`{=html}Mean   t=8.98, p=0.000`<br><br>`{=html}Mean   t=9.00, p=0.000`<br><br>`{=html}Mean
                                       diff=3.10 `<br>`{=html} Cohen's        diff=3.32 `<br>`{=html} Cohen's        diff=6.42 `<br>`{=html} Cohen's
                                       d=0.76`<br><br>`{=html}Distribution    d=0.82`<br><br>`{=html}Distribution    d=0.82`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.43, JS=0.52, WD=0.65             TVD=0.42, JS=0.52, WD=0.66             TVD=0.43, JS=0.52, WD=0.65

     89 qwen2.5-coder_0.5b_t0.25.csv   Mean (SD): 17.46 (1.73)                Mean (SD): 17.81 (1.86)                Mean (SD): 35.27 (2.59)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 13.00 /      `<br><br>`{=html} min/max 28.00 /
                                       23.00 `<br><br>`{=html} α raw=-0.04    24.00 `<br><br>`{=html} α raw=-0.02    43.00 `<br><br>`{=html} α raw=0.02
                                       `<br>`{=html} α std=-0.05              `<br>`{=html} α std=-0.01              `<br>`{=html} α std=0.02
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.65, p=0.000`<br><br>`{=html}Mean   t=8.13, p=0.000`<br><br>`{=html}Mean   t=7.81, p=0.000`<br><br>`{=html}Mean
                                       diff=2.57 `<br>`{=html} Cohen's        diff=3.13 `<br>`{=html} Cohen's        diff=5.70 `<br>`{=html} Cohen's
                                       d=0.60`<br><br>`{=html}Distribution    d=0.73`<br><br>`{=html}Distribution    d=0.71`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.21, JS=0.22, WD=0.45             TVD=0.24, JS=0.25, WD=0.52             TVD=0.22, JS=0.23, WD=0.47

     90 qwen2.5-coder_0.5b_t0.5.csv    Mean (SD): 17.66 (2.69)                Mean (SD): 17.90 (2.74)                Mean (SD): 35.56 (3.79)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 25.00 /
                                       24.00 `<br><br>`{=html} α raw=0.03     25.00 `<br><br>`{=html} α raw=0.08     46.00 `<br><br>`{=html} α raw=0.02
                                       `<br>`{=html} α std=0.03               `<br>`{=html} α std=0.07               `<br>`{=html} α std=0.03
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.84, p=0.000`<br><br>`{=html}Mean   t=8.01, p=0.000`<br><br>`{=html}Mean   t=8.02, p=0.000`<br><br>`{=html}Mean
                                       diff=2.77 `<br>`{=html} Cohen's        diff=3.22 `<br>`{=html} Cohen's        diff=5.98 `<br>`{=html} Cohen's
                                       d=0.61`<br><br>`{=html}Distribution    d=0.71`<br><br>`{=html}Distribution    d=0.72`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.23, JS=0.21, WD=0.46             TVD=0.25, JS=0.23, WD=0.54             TVD=0.24, JS=0.22, WD=0.50

     91 qwen2.5-coder_0.5b_t0.75.csv   Mean (SD): 17.76 (2.96)                Mean (SD): 17.71 (3.04)                Mean (SD): 35.47 (4.40)
                                       `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 9.00 / 25.00 `<br><br>`{=html} min/max 22.00 /
                                       27.00 `<br><br>`{=html} α raw=0.04     `<br><br>`{=html} α raw=0.06           47.00 `<br><br>`{=html} α raw=0.12
                                       `<br>`{=html} α std=0.05               `<br>`{=html} α std=0.06               `<br>`{=html} α std=0.12
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.99, p=0.000`<br><br>`{=html}Mean   t=7.40, p=0.000`<br><br>`{=html}Mean   t=7.78, p=0.000`<br><br>`{=html}Mean
                                       diff=2.87 `<br>`{=html} Cohen's        diff=3.02 `<br>`{=html} Cohen's        diff=5.89 `<br>`{=html} Cohen's
                                       d=0.62`<br><br>`{=html}Distribution    d=0.65`<br><br>`{=html}Distribution    d=0.70`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.24, JS=0.22, WD=0.48             TVD=0.23, JS=0.22, WD=0.50             TVD=0.23, JS=0.22, WD=0.49

     92 qwen2.5-coder_0.5b_t1.0.csv    Mean (SD): 17.53 (2.94)                Mean (SD): 17.75 (3.10)                Mean (SD): 35.28 (4.43)
                                       `<br><br>`{=html} min/max 9.00 / 25.00 `<br><br>`{=html} min/max 8.00 / 28.00 `<br><br>`{=html} min/max 24.00 /
                                       `<br><br>`{=html} α raw=-0.08          `<br><br>`{=html} α raw=0.01           48.00 `<br><br>`{=html} α raw=0.05
                                       `<br>`{=html} α std=-0.07              `<br>`{=html} α std=0.01               `<br>`{=html} α std=0.05
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.43, p=0.000`<br><br>`{=html}Mean   t=7.48, p=0.000`<br><br>`{=html}Mean   t=7.53, p=0.000`<br><br>`{=html}Mean
                                       diff=2.64 `<br>`{=html} Cohen's        diff=3.07 `<br>`{=html} Cohen's        diff=5.71 `<br>`{=html} Cohen's
                                       d=0.57`<br><br>`{=html}Distribution    d=0.66`<br><br>`{=html}Distribution    d=0.67`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.21, JS=0.19, WD=0.44             TVD=0.22, JS=0.21, WD=0.51             TVD=0.21, JS=0.20, WD=0.48

     93 qwen2.5-coder_0.5b_t1.2.csv    Mean (SD): 17.58 (2.96)                Mean (SD): 17.82 (3.37)                Mean (SD): 35.40 (4.49)
                                       `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 23.00 /
                                       25.00 `<br><br>`{=html} α raw=-0.11    27.00 `<br><br>`{=html} α raw=0.13     47.00 `<br><br>`{=html} α raw=0.03
                                       `<br>`{=html} α std=-0.12              `<br>`{=html} α std=0.13               `<br>`{=html} α std=0.02
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.54, p=0.000`<br><br>`{=html}Mean   t=7.51, p=0.000`<br><br>`{=html}Mean   t=7.67, p=0.000`<br><br>`{=html}Mean
                                       diff=2.69 `<br>`{=html} Cohen's        diff=3.13 `<br>`{=html} Cohen's        diff=5.82 `<br>`{=html} Cohen's
                                       d=0.58`<br><br>`{=html}Distribution    d=0.66`<br><br>`{=html}Distribution    d=0.68`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.21, JS=0.20, WD=0.45             TVD=0.24, JS=0.24, WD=0.52             TVD=0.22, JS=0.22, WD=0.48

     94 qwen2.5-coder_0.5b_t1.25.csv   Mean (SD): 17.40 (3.41)                Mean (SD): 17.47 (3.06)                Mean (SD): 34.87 (4.12)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 26.00 /
                                       25.00 `<br><br>`{=html} α raw=0.08     25.00 `<br><br>`{=html} α raw=-0.01    43.00 `<br><br>`{=html} α raw=-0.21
                                       `<br>`{=html} α std=0.10               `<br>`{=html} α std=0.01               `<br>`{=html} α std=-0.21
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.45, p=0.001`<br><br>`{=html}Mean   t=4.15, p=0.000`<br><br>`{=html}Mean   t=5.10, p=0.000`<br><br>`{=html}Mean
                                       diff=2.51 `<br>`{=html} Cohen's        diff=2.78 `<br>`{=html} Cohen's        diff=5.29 `<br>`{=html} Cohen's
                                       d=0.43`<br><br>`{=html}Distribution    d=0.49`<br><br>`{=html}Distribution    d=0.48`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.19, JS=0.21, WD=0.42             TVD=0.19, JS=0.19, WD=0.46             TVD=0.19, JS=0.19, WD=0.44

     95 qwen2.5-coder_0.5b_t1.4.csv    Mean (SD): 17.93 (3.07)                Mean (SD): 18.04 (3.31)                Mean (SD): 35.97 (4.43)
                                       `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 7.00 / 27.00 `<br><br>`{=html} min/max 24.00 /
                                       `<br><br>`{=html} α raw=-0.07          `<br><br>`{=html} α raw=0.03           51.00 `<br><br>`{=html} α raw=-0.06
                                       `<br>`{=html} α std=-0.07              `<br>`{=html} α std=0.03               `<br>`{=html} α std=-0.06
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.34, p=0.000`<br><br>`{=html}Mean   t=8.07, p=0.000`<br><br>`{=html}Mean   t=8.43, p=0.000`<br><br>`{=html}Mean
                                       diff=3.03 `<br>`{=html} Cohen's        diff=3.36 `<br>`{=html} Cohen's        diff=6.39 `<br>`{=html} Cohen's
                                       d=0.65`<br><br>`{=html}Distribution    d=0.71`<br><br>`{=html}Distribution    d=0.75`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.21, WD=0.51             TVD=0.24, JS=0.24, WD=0.56             TVD=0.23, JS=0.22, WD=0.53

     96 qwen2.5-coder_0.5b_t1.5.csv    Mean (SD): 17.39 (3.29)                Mean (SD): 18.00 (3.42)                Mean (SD): 35.39 (4.78)
                                       `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 21.00 /
                                       `<br><br>`{=html} α raw=0.06           `<br><br>`{=html} α raw=0.17           51.00 `<br><br>`{=html} α raw=0.12
                                       `<br>`{=html} α std=0.06               `<br>`{=html} α std=0.17               `<br>`{=html} α std=0.12
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=5.95, p=0.000`<br><br>`{=html}Mean   t=7.92, p=0.000`<br><br>`{=html}Mean   t=7.59, p=0.000`<br><br>`{=html}Mean
                                       diff=2.49 `<br>`{=html} Cohen's        diff=3.32 `<br>`{=html} Cohen's        diff=5.81 `<br>`{=html} Cohen's
                                       d=0.52`<br><br>`{=html}Distribution    d=0.70`<br><br>`{=html}Distribution    d=0.68`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.20, JS=0.19, WD=0.42             TVD=0.25, JS=0.24, WD=0.55             TVD=0.22, JS=0.21, WD=0.48

     97 qwen2.5-coder_0.5b_t1.6.csv    Mean (SD): 17.54 (3.21)                Mean (SD): 18.02 (3.24)                Mean (SD): 35.56 (4.53)
                                       `<br><br>`{=html} min/max 9.00 / 25.00 `<br><br>`{=html} min/max 9.00 / 26.00 `<br><br>`{=html} min/max 25.00 /
                                       `<br><br>`{=html} α raw=0.00           `<br><br>`{=html} α raw=-0.01          47.00 `<br><br>`{=html} α raw=-0.02
                                       `<br>`{=html} α std=-0.00              `<br>`{=html} α std=-0.02              `<br>`{=html} α std=-0.02
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.34, p=0.000`<br><br>`{=html}Mean   t=8.07, p=0.000`<br><br>`{=html}Mean   t=7.87, p=0.000`<br><br>`{=html}Mean
                                       diff=2.64 `<br>`{=html} Cohen's        diff=3.34 `<br>`{=html} Cohen's        diff=5.98 `<br>`{=html} Cohen's
                                       d=0.56`<br><br>`{=html}Distribution    d=0.71`<br><br>`{=html}Distribution    d=0.70`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.20, JS=0.19, WD=0.44             TVD=0.23, JS=0.23, WD=0.56             TVD=0.21, JS=0.21, WD=0.50

     98 qwen2.5-coder_0.5b_t1.7.csv    Mean (SD): 17.89 (3.05)                Mean (SD): 18.01 (3.35)                Mean (SD): 35.91 (4.64)
                                       `<br><br>`{=html} min/max 9.00 / 25.00 `<br><br>`{=html} min/max 9.00 / 29.00 `<br><br>`{=html} min/max 23.00 /
                                       `<br><br>`{=html} α raw=-0.16          `<br><br>`{=html} α raw=0.13           50.00 `<br><br>`{=html} α raw=0.05
                                       `<br>`{=html} α std=-0.17              `<br>`{=html} α std=0.13               `<br>`{=html} α std=0.05
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.26, p=0.000`<br><br>`{=html}Mean   t=7.98, p=0.000`<br><br>`{=html}Mean   t=8.30, p=0.000`<br><br>`{=html}Mean
                                       diff=3.00 `<br>`{=html} Cohen's        diff=3.33 `<br>`{=html} Cohen's        diff=6.33 `<br>`{=html} Cohen's
                                       d=0.64`<br><br>`{=html}Distribution    d=0.70`<br><br>`{=html}Distribution    d=0.74`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.21, JS=0.22, WD=0.50             TVD=0.25, JS=0.24, WD=0.55             TVD=0.23, JS=0.23, WD=0.53

     99 qwen2.5-coder_0.5b_t1.75.csv   Mean (SD): 17.33 (3.09)                Mean (SD): 17.27 (3.73)                Mean (SD): 34.60 (4.37)
                                       `<br><br>`{=html} min/max 12.00 /      `<br><br>`{=html} min/max 11.00 /      `<br><br>`{=html} min/max 26.00 /
                                       24.00 `<br><br>`{=html} α raw=-0.15    26.00 `<br><br>`{=html} α raw=0.21     46.00 `<br><br>`{=html} α raw=-0.18
                                       `<br>`{=html} α std=-0.07              `<br>`{=html} α std=0.14               `<br>`{=html} α std=-0.20
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=3.61, p=0.001`<br><br>`{=html}Mean   t=3.33, p=0.002`<br><br>`{=html}Mean   t=4.69, p=0.000`<br><br>`{=html}Mean
                                       diff=2.44 `<br>`{=html} Cohen's        diff=2.58 `<br>`{=html} Cohen's        diff=5.02 `<br>`{=html} Cohen's
                                       d=0.42`<br><br>`{=html}Distribution    d=0.45`<br><br>`{=html}Distribution    d=0.46`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.22, WD=0.41             TVD=0.19, JS=0.23, WD=0.43             TVD=0.20, JS=0.22, WD=0.42

    100 qwen2.5-coder_0.5b_t1.8.csv    Mean (SD): 17.82 (3.12)                Mean (SD): 17.98 (3.35)                Mean (SD): 35.80 (4.54)
                                       `<br><br>`{=html} min/max 9.00 / 27.00 `<br><br>`{=html} min/max 9.00 / 26.00 `<br><br>`{=html} min/max 26.00 /
                                       `<br><br>`{=html} α raw=-0.08          `<br><br>`{=html} α raw=0.08           51.00 `<br><br>`{=html} α raw=-0.02
                                       `<br>`{=html} α std=-0.08              `<br>`{=html} α std=0.08               `<br>`{=html} α std=-0.01
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.05, p=0.000`<br><br>`{=html}Mean   t=7.91, p=0.000`<br><br>`{=html}Mean   t=8.18, p=0.000`<br><br>`{=html}Mean
                                       diff=2.92 `<br>`{=html} Cohen's        diff=3.30 `<br>`{=html} Cohen's        diff=6.22 `<br>`{=html} Cohen's
                                       d=0.62`<br><br>`{=html}Distribution    d=0.70`<br><br>`{=html}Distribution    d=0.73`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.21, WD=0.49             TVD=0.23, JS=0.23, WD=0.55             TVD=0.23, JS=0.22, WD=0.52

    101 qwen2.5-coder_0.5b_t1.9.csv    Mean (SD): 17.77 (3.20)                Mean (SD): 18.09 (2.97)                Mean (SD): 35.86 (4.13)
                                       `<br><br>`{=html} min/max 9.00 / 28.00 `<br><br>`{=html} min/max 9.00 / 25.00 `<br><br>`{=html} min/max 23.00 /
                                       `<br><br>`{=html} α raw=-0.05          `<br><br>`{=html} α raw=-0.25          46.00 `<br><br>`{=html} α raw=-0.27
                                       `<br>`{=html} α std=-0.04              `<br>`{=html} α std=-0.25              `<br>`{=html} α std=-0.27
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=6.90, p=0.000`<br><br>`{=html}Mean   t=8.37, p=0.000`<br><br>`{=html}Mean   t=8.35, p=0.000`<br><br>`{=html}Mean
                                       diff=2.88 `<br>`{=html} Cohen's        diff=3.41 `<br>`{=html} Cohen's        diff=6.28 `<br>`{=html} Cohen's
                                       d=0.61`<br><br>`{=html}Distribution    d=0.74`<br><br>`{=html}Distribution    d=0.75`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.20, JS=0.20, WD=0.48             TVD=0.24, JS=0.23, WD=0.57             TVD=0.22, JS=0.22, WD=0.52

    102 qwen2.5-coder_0.5b_t2.0.csv    Mean (SD): 17.86 (3.14)                Mean (SD): 17.82 (3.27)                Mean (SD): 35.68 (4.50)
                                       `<br><br>`{=html} min/max 7.00 / 26.00 `<br><br>`{=html} min/max 10.00 /      `<br><br>`{=html} min/max 21.00 /
                                       `<br><br>`{=html} α raw=-0.07          26.00 `<br><br>`{=html} α raw=0.02     47.00 `<br><br>`{=html} α raw=-0.04
                                       `<br>`{=html} α std=-0.07              `<br>`{=html} α std=0.02               `<br>`{=html} α std=-0.03
                                       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:       `<br><br>`{=html}Prop Validated:
                                       1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:  1.00`<br><br>`{=html}t-test vs human:
                                       t=7.14, p=0.000`<br><br>`{=html}Mean   t=7.57, p=0.000`<br><br>`{=html}Mean   t=8.04, p=0.000`<br><br>`{=html}Mean
                                       diff=2.96 `<br>`{=html} Cohen's        diff=3.14 `<br>`{=html} Cohen's        diff=6.10 `<br>`{=html} Cohen's
                                       d=0.63`<br><br>`{=html}Distribution    d=0.67`<br><br>`{=html}Distribution    d=0.72`<br><br>`{=html}Distribution
                                       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}       distance vs human: `<br>`{=html}
                                       TVD=0.22, JS=0.21, WD=0.49             TVD=0.23, JS=0.22, WD=0.52             TVD=0.22, JS=0.21, WD=0.51
  ---------------------------------------------------------------------------------------------------------------------------------------------------------
:::
:::::

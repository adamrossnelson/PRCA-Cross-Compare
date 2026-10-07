# PRCA-Cross-Compare

A Quarto document that compares PRCA (Personal Report of Communication Apprehension) Survey Responder survey results across multiple model-generated data files. See [SurveyResponderData](https://github.com/adamrossnelson/SurveyResponderData) for more information.

## What it does

`index.qmd` loads a list of PRCA result CSVs, and builds a summary table reporting the mean, standard deviation, min/max, Cronbach's alpha (raw and standardized), and other metrics for use in comparing the LLM generated responses to human generated responses for three scales:

- Group CA
- Interpersonal CA
- Both Group and Interpersonal CA

## Requirements

- [Quarto](https://quarto.org/)
- Python with `pandas`, `numpy`, and `scipy`

## Usage

```bash
quarto render index.qmd
```

Or preview it live:

```bash
quarto preview index.qmd
```

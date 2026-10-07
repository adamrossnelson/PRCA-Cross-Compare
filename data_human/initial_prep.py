import pandas as pd

df = pd.read_csv('PRCA_October 7, 2026_06.16.csv', skiprows=[1,2])

print(df['Duration (in seconds)'].describe())

print(len(df))

print(len(df[df['Duration (in seconds)'] > 60]))

print(df.columns)

df = df[['StartDate', 'EndDate','Duration (in seconds)',
    'ResponseId', 'Q1', 'Q2', 'Q3', 'Q4',
    'Q5', 'Q6', 'Q13', 'Q14', 'Q15', 'Q16',
    'Q17', 'Q18']]

print(df.columns)

df = df.rename(
    columns={
        "Q1": "prca01",
        "Q2": "prca02",
        "Q3": "prca03",
        "Q4": "prca04",
        "Q5": "prca05",
        "Q6": "prca06",
        "Q13": "prca13",
        "Q14": "prca14",
        "Q15": "prca15",
        "Q16": "prca16",
        "Q17": "prca17",
        "Q18": "prca18",
    }
)

print(df.columns)

print(((df['prca01'] == 1) & (df['prca02'] == 1)).sum())
print(((df['prca01'] == 1) & (df['prca04'] == 1)).sum())

print(((df['prca13'] == 1) & (df['prca14'] == 1)).sum())
print(((df['prca15'] == 1) & (df['prca16'] == 1)).sum())

df.to_csv('human_responses.csv', index=False)

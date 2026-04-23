# Falcon Violation Learning Mapper

This Streamlit app is a small learning system for mapping new MVR descriptions to Falcon standards.

## What it does

For every new description, it:

1. normalizes the text
2. checks if the same input was already learned from user feedback
3. otherwise ranks Falcon standards using two metrics
   - cosine similarity on semantic text vectors
   - fuzzy token similarity
4. combines both scores into a final score
5. returns the top 2 Falcon matches
6. lets the user accept, reject, or manually correct the result
7. stores accepted feedback in a local SQLite database so future searches improve over time

## Output format

The suggestion is formatted like:

```text
61190 - DEFECTIVE/NO HEADLAMPS
```

The app uses:
- `SVCCODE - description` when `SVCCODE` exists
- otherwise `violation - description`

## Feedback learning

Feedback is stored in:

- `/mnt/data/falcon_feedback.db`

Accepted feedback creates a learned alias for that normalized input. Next time the same or very similar description comes in, the app can return the learned Falcon standard directly.

## Required files

- `Extracted VIOLATION CODE Mapping to FALCON CODES.xlsx`
- `Target Fields - Standard Falcon Driver-Violations Code.xlsx`

Keep them in `/mnt/data` or upload them from the sidebar.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Batch mode

Upload a CSV or Excel file with a description column such as:
- `description`
- `violation description`
- `mvr description`

The output contains top 1 and top 2 suggestions with cosine, fuzzy, and final scores.

# Automated machine learning pipeline

projeto2.py orchestrates extraction, cleaning, preprocessing, random forest training, evaluation and inference. Modules under pipeline/ share a scaler, column layout and model so new observations use the training representation.

## Run locally

```sh
python -m pip install -r requirements.txt
python projeto2.py
```

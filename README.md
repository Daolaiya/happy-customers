# Happy Customers

Machine learning analysis for predicting customer happiness from survey responses (Happy Customers challenge).

## Project layout

```
HC/
├── notebook_hc.ipynb   # Main analysis notebook
├── data_raw.csv        # Public training subset (126 rows)
├── models/             # Saved model pickles (created when notebook runs)
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter notebook notebook_hc.ipynb
```

`data_raw.csv` is already included. The Google Drive link in the challenge brief is optional if you use this copy.

## Usage

1. Open `notebook_hc.ipynb`.
2. Run all cells from top to bottom (runtime roughly 5–10 minutes depending on hardware).
3. The modelling section saves tuned models to `models/`. The feature-importance section loads those files.

## Challenge brief

**Background:**
- We are one of the fastest growing startups in the logistics and delivery domain. We work with several partners and make on-demand delivery to our customers. During the COVID-19 pandemic, we are facing several different challenges and every day we are trying to address these challenges.
- We thrive on making our customers happy. As a growing startup, with a global expansion strategy we know that we need to make our customers happy and the only way to do that is to measure how happy each customer is. If we can predict what makes our customers happy or unhappy, we can then take necessary actions.
- Getting feedback from customers is not easy either, but we do our best to get constant feedback from our customers. This is a crucial function to improve our operations across all levels.
- We recently did a survey to a select customer cohort. You are presented with a subset of this data. We will be using the remaining data as a private test set.

**Data description:**
- Y = target (0 = unhappy, 1 = happy)
- X1 = my order was delivered on time
- X2 = contents of my order were as I expected
- X3 = I ordered everything I wanted to order
- X4 = I paid a good price for my order
- X5 = I am satisfied with my courier
- X6 = the app makes ordering easy for me

Attributes X1–X6 are Likert scores from 1 (less) to 5 (more).

**Download data (optional):** https://drive.google.com/open?id=1KWE3J0uU_sFIJnZ74Id3FDBcejELI7FD

**Goals:**
- Predict whether a customer is happy from survey answers.
- Reach 73% accuracy or explain why the approach is still valuable.

**Bonus:**
- Feature selection: identify a minimal question set for future surveys.

## Results summary

| Item | Finding |
|------|---------|
| Data split | 80/20 stratified holdout (`random_state=0`): 100 train / 26 test |
| Best tuned models | XGBoost and SGD (~65% test accuracy); KNN, SVM, and RF ~62%; decision tree ~50% |
| 73% metric | Final CV-tuned models do **not** reach 73% on the 26-row holdout; LazyPredict shows an untuned Perceptron near 73% on the same split, but it is not cross-validated |
| Feature importance | X1 and X3 are consistently strong; X2 is often weakest (candidate to drop from a shorter survey) |
| Evaluation | Hyperparameters tuned with 5-fold CV on training data only; test set used once for final reporting |

See the **Inference** and **Feature importance** sections in the notebook for full discussion.

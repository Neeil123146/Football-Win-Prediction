# Football Win Prediction System Using Machine Learning

**Final Year B.E. Computer Engineering Capstone Project**  
*Academic Year: 2025–2026*

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Dataset](#3-dataset)
4. [Feature Engineering](#4-feature-engineering)
5. [Machine Learning Models](#5-machine-learning-models)
6. [Evaluation](#6-evaluation)
7. [Live Pre-Match Prediction](#7-live-pre-match-prediction)
8. [Web Application](#8-web-application)
9. [Installation](#9-installation)
10. [Environment Variables](#10-environment-variables)
11. [Usage](#11-usage)
12. [Project Structure](#12-project-structure)
13. [Limitations](#13-limitations)
14. [Future Scope](#14-future-scope)

---

## 1. Project Overview

The **Football Win Prediction System Using Machine Learning** is an engineering and data science capstone project designed to predict the outcome of English Premier League (EPL) association football fixtures prior to kickoff. Football match forecasting presents a classic multi-class classification challenge involving three distinct outcomes: **Home Win (`H`)**, **Draw (`D`)**, and **Away Win (`A`)**.

Predicting football outcomes is notoriously difficult due to:
- **High Stochasticity**: Low-scoring dynamics where a single deflection, referee decision, or red card can dictate the outcome.
- **Lookahead Bias (Target Leakage)**: Common pitfalls in naive ML pipelines where future match statistics or contemporaneous match events leak into training vectors.
- **Draw Class Under-Prediction**: Standard multi-class classifiers frequently minimize global empirical loss by ignoring the minority draw outcome (~23–25% of all matches), resulting in degenerate models that predict only wins and losses.

This project delivers an end-to-end, mathematically rigorous machine learning system that enforces strict chronological validation, generates a deterministic 53-feature pre-match schema, implements probability calibration, provides transparent model interpretability ("Why this prediction?"), and bridges the offline model with live pre-match data feeds through an interactive Flask web analytics dashboard.

---

## 2. Objectives

The primary academic and technical objectives of this project are:

1. **Strict Elimination of Data Leakage**: Enforce chronological ordering such that any feature vector for match $N$ is derived strictly from historical matches $1$ through $N-1$ completed prior to kickoff.
2. **Comprehensive 10-Season Historical Modeling**: Curate, clean, and process 10 full seasons of Premier League match records (3,800 matches) spanning 34 unique clubs.
3. **Rigorous Feature Engineering (53 Pre-Match Predictors)**: Construct multi-faceted indicators covering short-term rolling momentum, lifetime records, home/away venue splits, historical head-to-head dynamics, dynamic sequential Elo ratings, and current season standings.
4. **Resolution of Draw-Class Under-Prediction**: Formulate balanced training techniques (class weighting, sample adjustments) to restore sensitivity to the draw class while preserving overall predictive accuracy.
5. **Multi-Criteria Model Selection via Expanding Temporal CV**: Evaluate diverse algorithms (Logistic Regression, Random Forest, XGBoost, and Support Vector Machines) using an expanding-window time-series cross-validation protocol that reflects real-world operational forecasting.
6. **Empirical Probability Calibration**: Measure prediction reliability using multi-class Log Loss and the multi-class Brier score, ensuring returned probability vectors represent true statistical likelihoods rather than arbitrary classification confidence.
7. **Transparent Explainability ("Why this prediction?")**: Derive post-inference contributing factors rooted directly in underlying pre-match features, presenting clear statistical rationale with explicit non-causal disclaimers.
8. **Live Pre-Match Prediction Integration**: Ingest current completed and scheduled fixture feeds from real-world football data APIs, normalising club identities to canonical schema and generating identical 53-feature vectors for real-time pre-match inference.
9. **Production-Ready Web Dashboard**: Deliver a glassmorphic user interface supporting live and offline modes, betting odds display, data provenance tracking, and model comparison diagnostics.

---

## 3. Dataset

The historical foundation of the system is derived from official Premier League match archives hosted at **football-data.co.uk**:

* **League**: English Premier League (EPL, Division `E0`).
* **Temporal Breadth**: **10 complete historical seasons** (2015–16 through 2024–25).
* **Total Matches**: Exactly **3,800 matches** ($10 \text{ seasons} \times 380 \text{ matches/season}$).
* **Club Representation**: Exactly **34 unique clubs** represented across the historical dataset.
  > *Note on Terminology*: The Premier League consists of 20 clubs in any individual season. Over the 10-season historical window (2015–2025), promotions from and relegations to the EFL Championship resulted in 34 unique clubs appearing in the dataset:
  >
  > *Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Cardiff, Chelsea, Crystal Palace, Everton, Fulham, Huddersfield, Hull, Ipswich, Leeds, Leicester, Liverpool, Luton, Man City, Man United, Middlesbrough, Newcastle, Norwich, Nott'm Forest, Sheffield United, Southampton, Stoke, Sunderland, Swansea, Tottenham, Watford, West Brom, West Ham, Wolves.*
* **Data Granularity**: Match-result-based historical data, capturing match date, participating clubs, half-time and full-time scores, shots, shots on target, corners, fouls, and disciplinary cards.
* **Class Distribution across the 3,800 Matches**:
  - **Home Win (`H`)**: 1,691 matches (**44.50%**)
  - **Away Win (`A`)**: 1,223 matches (**32.18%**)
  - **Draw (`D`)**: 886 matches (**23.32%**)

---

## 4. Feature Engineering

The system computes an exact **53-feature pre-match vector** for every fixture. All 53 features are derived exclusively from events occurring prior to match kickoff.

### 53-Feature Taxonomy

| Group | Feature Name | Description | Pre-Match Derivation |
|---|---|---|---|
| **1. Recent 5-Match Form** (8) | `Home_Recent_Pts`, `Away_Recent_Pts` | Points accumulated across last 5 matches | 3 for W, 1 for D, 0 for L |
| | `Home_Recent_WinRate`, `Away_Recent_WinRate` | Win rate over last 5 matches | Wins / 5 |
| | `Home_Recent_GF_PG`, `Away_Recent_GF_PG` | Goals scored per game in last 5 matches | GF / 5 |
| | `Home_Recent_GA_PG`, `Away_Recent_GA_PG` | Goals conceded per game in last 5 matches | GA / 5 |
| **2. Recent Form Differentials** (2) | `Diff_Recent_Pts` | Difference in recent points | Home - Away |
| | `Diff_Recent_WinRate` | Difference in recent win rate | Home - Away |
| **3. Lifetime Overall Records** (10) | `Home_Overall_MP`, `Away_Overall_MP` | Cumulative matches played in historical dataset | Count of prior matches |
| | `Home_Overall_PPG`, `Away_Overall_PPG` | Historical points per game | Total Pts / MP |
| | `Home_Overall_GF_PG`, `Away_Overall_GF_PG` | Historical goals scored per game | Total GF / MP |
| | `Home_Overall_GA_PG`, `Away_Overall_GA_PG` | Historical goals conceded per game | Total GA / MP |
| | `Home_Overall_GD`, `Away_Overall_GD` | Historical cumulative goal difference | GF - GA |
| **4. Lifetime Differentials** (2) | `Diff_PPG` | Historical PPG difference | Home PPG - Away PPG |
| | `Diff_Overall_GD` | Historical goal difference margin | Home GD - Away GD |
| **5. Lifetime Venue Splits** (6) | `Home_Home_WR` | Home club's historical home win rate | Home Wins / Home MP |
| | `Home_Home_GF_PG`, `Home_Home_GA_PG` | Home club's home goals scored/conceded PG | Home GF / MP, Home GA / MP |
| | `Away_Away_WR` | Away club's historical road win rate | Away Wins / Away MP |
| | `Away_Away_GF_PG`, `Away_Away_GA_PG` | Away club's road goals scored/conceded PG | Road GF / MP, Road GA / MP |
| **6. Season Venue Splits** (6) | `Home_Season_Home_WR` | Current season home win rate | Current season Home Wins / MP |
| | `Home_Season_Home_GF_PG`, `Home_Season_Home_GA_PG` | Current season home goals PG | Current season Home GF/GA PG |
| | `Away_Season_Away_WR` | Current season away win rate | Current season Road Wins / MP |
| | `Away_Season_Away_GF_PG`, `Away_Season_Away_GA_PG` | Current season road goals PG | Current season Road GF/GA PG |
| **7. Head-to-Head Dynamics** (5) | `H2H_Prior_Matches` | Prior historical encounters between clubs | Direct prior meetings count |
| | `H2H_HomeWins`, `H2H_Draws`, `H2H_AwayWins` | Prior H2H match outcome counts | Specific H2H results |
| | `H2H_GoalDiff` | Cumulative goal difference in prior H2H | Home GF - Away GF in H2H |
| **8. Dynamic Sequential Elo** (3) | `Home_Elo`, `Away_Elo` | Sequential Elo rating before kickoff | Starting $R_0 = 1500$, $K = 20$ |
| | `Elo_Diff` | Rating differential | `Home_Elo` - `Away_Elo` |
| **9. Current Season Standings** (11) | `Home_Season_Rank`, `Away_Season_Rank` | League table position (1 to 20) | Pre-kickoff standings |
| | `Home_Season_Pts`, `Away_Season_Pts` | Current season accumulated league points | Standings points |
| | `Home_Season_GD`, `Away_Season_GD` | Current season goal difference | Standings GD |
| | `Home_Season_MP`, `Away_Season_MP` | Matches played in current season | Standings MP |
| | `Diff_Rank` | Table position advantage | Away Rank - Home Rank |
| | `Diff_Season_Pts`, `Diff_Season_GD` | Season points and GD differentials | Home - Away |

### How Target Leakage is Strictly Prevented

Data leakage (or lookahead bias) is the single most pervasive flaw in machine learning sports modeling. It occurs when features incorporate information that was not yet known at prediction time. In this system, leakage is prevented through architectural guarantees:

1. **Pre-Kickoff Cutoff**: When engineering features for match $m_{i}$ at date $T_{i}$, the system filters the chronological match history to strictly include only matches where $\text{Date} < T_{i}$ (or earlier match indices on the same date).
2. **Post-Match State Mutation**: Running team accumulators (form queues, goal tallies, standings, and Elo updates) are updated **only after** the match's features have been vectorized and stored.
3. **No In-Match Feature Contamination**: In-game statistics such as full-time goals (`FTHG`, `FTAG`), half-time scores (`HTHG`, `HTAG`), in-game shots (`HS`, `AS`), and cards are **never** features; they are utilized solely as historical outcomes after match completion.
4. **StandardScaler Isolation**: The `StandardScaler` is fitted **strictly on the historical training set** (2015–16 to 2021–22) and persisted to disk. The validation and test sets (and live fixtures) are transformed using the locked scaler parameters without refitting.

---

## 5. Machine Learning Models

Four distinct machine learning algorithms representing different inductive biases were tuned and evaluated:

1. **Logistic Regression (Multinomial with L2 Regularization)**: Serves as the linear probabilistic baseline, providing well-behaved log-odds under convex optimization.
2. **Random Forest Classifier**: Non-linear bagging ensemble of decision trees. Captures non-linear thresholds and complex feature interactions without overfitting noisy individual matches.
3. **XGBoost (Extreme Gradient Boosting)**: Gradient-boosted decision tree framework optimizing multi-class log-loss via second-order Taylor approximations.
4. **Support Vector Classifier (SVM with RBF Kernel)**: Non-linear kernel classifier with Platt scaling (`probability=True`) to map margin distances to calibrated probabilities.

### Temporal Validation Methodology

Traditional random $k$-fold cross-validation or randomized train-test splits violate temporal ordering by training on future fixtures to predict past matches. To maintain academic rigor, the project implemented an **expanding-window chronological validation protocol**:

```
Fold 1: Train [2015-16 to 2018-19] (1,520 matches) -> Validate on [2019-20] (380 matches)
Fold 2: Train [2015-16 to 2019-20] (1,900 matches) -> Validate on [2020-21] (380 matches)
Fold 3: Train [2015-16 to 2020-21] (2,280 matches) -> Validate on [2021-22] (380 matches)
Fold 4: Train [2015-16 to 2021-22] (2,660 matches) -> Validate on [2022-23] (380 matches)
```

The final two seasons (**2023–24 and 2024–25, 760 matches**) were held out as a completely untouched test set and **were never used for model selection, hyperparameter tuning, or calibration decisions**.

### Why Random Forest Was Selected as the Final Champion Model

Model selection was governed by a multi-criteria decision index evaluated across all 4 expanding validation folds:

$$\text{Score} = w_1 (1 - \text{LogLoss}_{\text{norm}}) + w_2 (1 - \text{Brier}_{\text{norm}}) + w_3 \text{MacroF1} + w_4 \text{DrawRecall} + w_5 \text{Accuracy} + w_6 \text{Stability}$$

* **Random Forest achieved the highest composite selection score (0.9055)**.
* It delivered the **highest temporal cross-validation accuracy (50.39%)** and the **lowest temporal variance ($\pm 1.85\%$)** across folds.
* It achieved the **lowest mean validation Log Loss (1.0232)** and **lowest Multiclass Brier Score (0.6092)**.
* Unlike XGBoost (which overfit training matches: 99.7% train acc vs 47.0% CV acc, Log Loss 1.3758), Random Forest generalized cleanly without extreme probability overconfidence.
* Consequently, **Random Forest** was locked as the production inference model.

---

## 6. Evaluation

All reported metrics reflect actual, verified evaluations computed on un-modified historical data.

### 1. Expanding-Window Temporal Cross-Validation Results (4 Validation Folds)

*Across validation folds covering seasons 2019–20, 2020–21, 2021–22, and 2022–23:*

| Model | Mean Accuracy | Accuracy Std | Mean Macro F1 | Mean Log Loss | Mean Brier Score | Mean Draw Recall | Composite Rank |
|---|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | **50.39%** | **$\pm$ 1.85%** | **0.4435** | **1.0232** | **0.6092** | **15.72%** | **1 (Score: 0.9055)** |
| Logistic Regression | 47.17% | $\pm$ 2.09% | 0.4299 | 1.0556 | 0.6273 | 20.23% | 2 (Score: 0.7332) |
| SVM (RBF Kernel) | 45.20% | $\pm$ 3.02% | 0.4239 | 1.0255 | 0.6119 | 25.42% | 3 (Score: 0.6646) |
| XGBoost | 47.04% | $\pm$ 2.93% | 0.3944 | 1.3758 | 0.7348 | 10.03% | 4 (Score: 0.0609) |

*Fold-by-Fold Accuracies:*
- **Random Forest**: Fold 1 = 52.11%, Fold 2 = 48.42%, Fold 3 = 52.37%, Fold 4 = 48.68%
- **Logistic Regression**: Fold 1 = 49.21%, Fold 2 = 47.63%, Fold 3 = 48.16%, Fold 4 = 43.68%
- **XGBoost**: Fold 1 = 48.95%, Fold 2 = 42.11%, Fold 3 = 49.47%, Fold 4 = 47.63%
- **SVM**: Fold 1 = 48.68%, Fold 2 = 42.89%, Fold 3 = 47.63%, Fold 4 = 41.58%

---

### 2. Final Untouched Test Set Evaluation (760 Matches, Seasons 2023–24 & 2024–25)

The final test set comprises 760 unseen fixtures. It was strictly evaluated once after all training and validation decisions were finalized.

* **Naive Baseline (Always predicting Home Win)**: $\frac{330}{760} = \mathbf{43.42\%}$

| Model | Test Accuracy | Lift over Naive | Macro F1 | Log Loss | Multiclass Brier Score | Draw Recall | Draw Precision |
|---|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | **50.66%** | **+7.24%** | **0.4441** | **0.9964** | **0.5925** | **14.29%** | **28.09%** |
| Logistic Regression | 49.08% | +5.66% | 0.4499 | 1.0144 | 0.6048 | 20.57% | 23.53% |
| XGBoost | 48.68% | +5.26% | 0.4240 | 1.1665 | 0.6682 | 16.00% | 31.46% |
| SVM | 48.55% | +5.13% | 0.4583 | 1.0101 | 0.6031 | 28.57% | 29.94% |

#### Random Forest Test Confusion Matrix (760 Matches)

| Actual \ Predicted | Predicted Away Win | Predicted Draw | Predicted Home Win | Total Actual | Class Recall | Class Precision |
|---|---|---|---|---|---|---|
| **Actual Away Win (`A`)** | **159** | 25 | 71 | 255 | 62.35% | 48.92% (159/325) |
| **Actual Draw (`D`)** | 76 | **25** | 74 | 175 | 14.29% | 28.09% (25/89) |
| **Actual Home Win (`H`)** | 90 | 39 | **201** | 330 | 60.91% | 58.09% (201/346) |
| **Total Predicted** | 325 | 89 | 346 | 760 | Overall Acc: **50.66%** | |

*Key Diagnostic Insight*: Standard unweighted models predict 0 draws (0.0% draw recall). Through balanced sample weights, the Random Forest model predicted 89 draws and correctly identified 25 draws in out-of-sample testing while achieving sub-1.0 test Log Loss (**0.9964**).

---

## 7. Live Pre-Match Prediction

### System Architecture

The live pre-match architecture establishes complete functional symmetry between historical model training and live deployment:

```
[ Historical Pipeline ]
Historical Match Data (10 Seasons, 3,800 Matches)
       │
       ▼
Chronological Feature Engineering (53 Pre-Match Features)
       │
       ▼
Model Training & Temporal CV Validation
       │
       ▼
Locked Champion Model (Random Forest) & Locked StandardScaler

──────────────────────────────────────────────────────────────────

[ Live Pre-Match Pipeline ]
Current Premier League Data (football-data.co.uk / football-data.org)
       │
       ▼
Canonical Team Normalization (34 Historical Clubs)
       │
       ▼
Pre-Match Feature Generator (Strict Pre-Match Cutoff: Date < Kickoff)
       │
       ▼
Exact Same 53 Pre-Match Feature Vector (Identical Ordering & Logic)
       │
       ▼
Existing Locked StandardScaler (No refitting)
       │
       ▼
Existing Locked Champion Model (Random Forest)
       │
       ▼
Calibrated Probabilities & "Why This Prediction?" Explainability
```

### Pre-Match Scope Clarification

> **CRITICAL DISTINCTION**: This system performs **live pre-match prediction** using up-to-the-minute league tables, recent form, and scheduled fixture metadata prior to kickoff.
>
> It does **not** perform in-game, in-play, or second-by-second live match tracking (e.g., ball possession, player GPS tracking, in-game momentum shifts). Once kickoff occurs, match conditions are frozen according to pre-match states.

---

## 8. Web Application & Interactive Prediction Platform

The system includes a production-grade web application built with a **Flask backend** and a modern **dark-mode glassmorphic frontend** (Vanilla CSS and JavaScript, no Tailwind dependencies).

### Core Platform Architecture & Pages

1. **Primary Analytics Dashboard (`/`)**:
   - **Live Status Ribbon**: Real-time operational indicator displaying active data provider (ESPN Premier League Calendar, Football-Data), cache state, and synchronization timestamp.
   - **Live Match Center & Upcoming Fixtures**: Displays upcoming Premier League fixtures across October 2026 and beyond (November 2026, custom date ranges). Clicking *"Predict This Match"* directs the user to the dedicated prediction page for that particular fixture.
   - **Dual Pipeline Switcher**: Live 53-Feature Mode vs Historical Locked State.
   - **Explainability & Model Diagnostics**: 53-feature provenance, historical test confusion matrix, and 760-match backtest report.

2. **Dedicated Match Prediction Page (`/predict/<fixture_id>`)**:
   - **Fixture Attribution**: Home and away club crests, names, venue, scheduled kickoff time with automatic UTC to client local timezone conversion, and Premier League badge.
   - **Pre-Match Status Badge**: Tagged with an authoritative `PRE-MATCH PREDICTION` indicator.
   - **Prominent Projected Outcome**: Highlights the most probable outcome (*Home Win*, *Draw*, or *Away Win*) with clear visual emphasis.
   - **Calibrated Multi-Class Probability Meters**: Displays separate meters for Home Win, Draw, and Away Win summing to approximately 100.0%.
   - **"Why This Prediction?" 4-Pillar Deep Dive**:
     - *Elo Rating & Team Strength*: Pre-match Elo ratings and differential.
     - *Recent Momentum & Form*: 5-game form string, win rates, and points.
     - *Venue Splits*: Home ground win % vs road trip win %, goals scored and conceded.
     - *Historical Head-to-Head*: Historical encounters, wins, draws, and goal differentials.
   - **Inspectable 53-Feature Vector**: Collapsible audit drawer exposing every single numerical feature value passed to the machine learning model.
   - **Confirmation Action**: *"Confirm & Save Prediction"* button that commits the prediction to permanent history.

3. **Persistent Prediction History & Verification Tracker (`/history`)**:
   - **KPI Summary Grid**: Total Predictions, Historical Accuracy %, Correct Outcomes, Incorrect Outcomes, and Pending Matches.
   - **Historical Accuracy Calculation**:
     $$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Correct Predictions} + \text{Incorrect Predictions}} \times 100$$
     *Excludes pending, postponed, and cancelled matches from the denominator to prevent statistical distortion. If 0 verified, safely displays "Pending Results" without dividing by zero.*
   - **Filter Tabs**: Filter by *All Predictions*, *Pending*, *Correct*, *Incorrect*, and *Completed Matches*.
   - **Automated & Manual Verification Controls**: Dedicated *"Refresh & Verify Results Now"* button alongside last sync timestamp and provider connection status.

### Prediction & Verification Lifecycle

```
UPCOMING MATCH
      │
      ▼
USER CONFIRMS PREDICTION  ──►  Saved to data/prediction_history.json (Thread-Safe Atomic Write)
      │
      ▼
MATCH KICKS OFF (PENDING)
      │
      ▼
MATCH IN PROGRESS — RESULT PENDING
      │
      ▼
PROVIDER CONFIRMS MATCH FINISHED
      │
      ▼
FETCH & VALIDATE OFFICIAL FINAL SCORE (Home Score vs Away Score)
      │
      ▼
DETERMINE ACTUAL OUTCOME (Home Goals > Away Goals = Home Win, etc.)
      │
      ▼
COMPARE PREDICTED OUTCOME WITH OFFICIAL OUTCOME
      │
      ├── Match ──► Status: CORRECT (is_correct = True)
      └── Differ ──► Status: INCORRECT (is_correct = False)
      (Postponed/Cancelled matches remain POSTPONED, never penalized as incorrect)
```

### Backend REST API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main football win prediction dashboard |
| `/predict/<fixture_id>` | `GET` | Dedicated match prediction interface (HTML or JSON via `?format=json`) |
| `/history` | `GET` | Prediction history and verification tracking dashboard |
| `/api/predictions/confirm` | `POST` | Confirms and saves prediction to persistent storage (with deduplication) |
| `/api/predictions/history` | `GET` | Retrieves all saved prediction records and summary accuracy metrics |
| `/api/predictions/verify` | `POST` | Reconciles pending predictions against official final scores |
| `/api/fixture/<fixture_id>` | `GET` | Retrieves fixture metadata and pre-match 53-feature prediction JSON |
| `/api/live/fixtures` | `GET` | Fetches live upcoming fixtures supporting October, November, and custom ranges |
| `/api/live/predict` | `POST` | Runs live 53-feature inference for any two clubs and match date |
| `/api/live/status` | `GET` | Diagnostic health status of providers, disk cache, and 53-feature engine |
| `/api/teams` | `GET` | List of all 34 canonical Premier League clubs in the dataset |
| `/api/team-stats/<team>` | `GET` | Pre-match Elo, form, goals, and win rate metrics for an individual club |
| `/api/models` | `GET` | Cross-model performance comparison metrics across Train, Val, Test |
| `/api/backtest` | `GET` | Chronological backtesting report across 760 test matches |

---

## 9. Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Football-Win-Prediction.git
cd Football-Win-Prediction
```

### Step 2: Create and Activate Virtual Environment
*On Windows (PowerShell):*
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*On Linux/macOS:*
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 10. Environment Variables

Create a local `.env` file by copying `.env.example`:
```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `FLASK_DEBUG` | No | `0` | Set to `1` for development auto-reloading; leave `0` in production. |
| `PORT` | No | `5000` | HTTP port on which Flask listens. |
| `HOST` | No | `127.0.0.1` | Network interface binding. |
| `FOOTBALL_DATA_API_KEY`| No | *None* | Optional API token for football-data.org. The default provider (`football-data.co.uk`) operates with **zero authentication required**. |

---

## 11. Usage

### Step 1: Execute Full Verification Suite
Verify that all ML artifacts, datasets, feature schemas, and web endpoints pass test validation:
```bash
python run_all_tests.py
```

### Step 2: Start the Web Application
```bash
python app.py
```
*Output:*
```
Starting Football Win Prediction System Flask Server...
Pre-Match Match Outcome Prediction Dashboard: http://127.0.0.1:5000
Debug Mode: False (set FLASK_DEBUG=1 to enable for development)
```

### Step 3: Open the Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Step 4: Generate a Prediction
1. **Option A (Upcoming Match)**: In the *Live Match Center*, click *"Predict Match"* on any scheduled fixture.
2. **Option B (Manual Selection)**: Select a **Home Club** and an **Away Club** from the dropdowns and click *"Run Pre-Match Prediction"*.

### Step 5: Interpret Results
- Inspect the predicted outcome (*Home Win*, *Draw*, or *Away Win*) and the corresponding probability distribution.
- Review the *Data Provenance Strip* to confirm whether live API or cached data was utilized.
- Read the *"Why This Prediction?"* panel to inspect contributing factors (Elo differentials, form momentum, venue records).

---

## 12. Project Structure

```
Football-Win-Prediction/
├── .env.example                      # Environment configuration template
├── .gitignore                         # Excludes secrets, venvs, cache, and logs
├── README.md                          # Comprehensive academic and technical documentation
├── requirements.txt                   # Production Python package dependencies
├── app.py                             # Flask web application & REST API server
├── run_all_tests.py                   # Master test runner executing all 8 validation suites
├── test_upgraded_platform.py          # Dedicated prediction page & history verification tests
├── test_comprehensive_suite.py        # End-to-end 24-test suite (ML, prediction, live, web)
├── verify_structure.py                # File architecture and model loading verification
├── inspect_53_features.py             # Schema parity and zero-leakage validator
├── audit_project.py                   # 5-point subsystem health check
├── verify_live_feature_generator.py   # Historical parity validator for live feature generator
├── verify_live_api_integration.py     # Live API endpoint and fallback test harness
├── verify_step7_ui.py                 # UI DOM elements and AJAX flow verification
├── data/
│   ├── raw/                           # 10 raw season CSVs (EPL_2015-16.csv to EPL_2024-25.csv)
│   ├── processed/                     # Cleaned matches (3,800 rows) and engineered features
│   ├── cache/                         # Local JSON disk cache with TTL expiry
│   └── prediction_history.json        # Persistent user prediction history & audit records
├── models/
│   ├── best_model.pkl                 # Locked champion model (Random Forest, 50.66% test acc)
│   ├── scaler.pkl                     # Locked StandardScaler (fitted strictly on training set)
│   ├── metadata.json                  # Schema definition, feature list, and temporal CV logs
│   ├── model_comparison.json          # Complete evaluation metrics across all 4 models
│   ├── feature_importance.csv         # Ranked feature importances from champion model
│   ├── logistic_regression.pkl        # Serialized Logistic Regression baseline
│   ├── random_forest.pkl              # Serialized Random Forest estimator
│   ├── xgboost.pkl                    # Serialized XGBoost model
│   └── svm.pkl                        # Serialized Support Vector Classifier
├── notebooks/
│   └── football_analysis.ipynb        # Exploratory data analysis and experimental logs
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Raw dataset ingestion and schema alignment
│   ├── data_preprocessing.py          # Data cleaning, timestamp parsing, and validation
│   ├── feature_engineering.py         # 53 pre-match feature pipeline and Elo rating engine
│   ├── train_models.py                # Chronological training and temporal CV execution
│   ├── evaluate_models.py             # Calibration diagnostics and metrics computation
│   ├── prediction.py                  # MatchPredictor inference engine and explainability
│   ├── prediction_history.py          # Persistent history manager and KPI accuracy calculator
│   ├── result_verifier.py             # Automated official match result verification engine
│   ├── team_crests.py                 # Club branding, colors, badges, and stadiums
│   ├── data/
│   │   ├── __init__.py
│   │   └── live_data_fetcher.py       # Live provider fetcher, team normalization, caching
│   └── features/
│       ├── __init__.py
│       └── live_feature_generator.py  # Real-time 53-feature derivation from live feeds
├── static/
│   ├── style.css                      # Premium dark-mode glassmorphic styling
│   ├── script.js                      # Client-side AJAX handling, tabs, odds, and rendering
│   └── images/                        # Confusion matrices, calibration and ROC curves
└── templates/
    ├── index.html                     # Responsive semantic HTML5 dashboard template
    ├── match_prediction.html          # Dedicated pre-match prediction page
    └── history.html                   # Prediction history & results verification tracker
```

---

## 13. Limitations

In accordance with academic honesty, several real-world limitations must be acknowledged:

1. **Inherent Stochasticity of Association Football**: Unlike higher-scoring sports such as basketball, association football features low goal counts (average ~2.7 goals/match). A substantial portion of variance is governed by unmodelable stochastic events (deflections, referee decisions, red cards, weather extremes). No statistical model can guarantee individual match outcomes.
2. **The Intrinsic Difficulty of the Draw Class**: Draws in football are notoriously challenging to classify because a draw is often not a planned objective, but an equilibrium outcome resulting from cancelled advantages. While our balanced training improved draw recall to 14.3% with 28.1% precision, draws remain the most difficult class to forecast.
3. **Absence of Tactical and Tracking Telemetry**: The historical dataset relies on match-level summary statistics. It does not incorporate player-level tracking (GPS, sprint velocity), in-depth tactical lineups, or expected goals ($xG$) shot coordinates, which were unavailable across the entire 10-season historical span.
4. **Third-Party API Dependency & Stale Data**: Live pre-match data depends on third-party provider availability. While the system implements disk caching and automatic stale fallback, upstream provider outages or delayed fixture updates can affect feature freshness.
5. **Promoted Club Sample Scarcity**: Promoted clubs (e.g., Luton Town, Ipswich Town) have fewer prior matches in the 10-season historical dataset, requiring baseline prior adjustments during early seasonal fixtures.

---

## 14. Future Scope

Potential technical extensions for future research include:

1. **Expansion to Multi-League Datasets**: Generalizing the chronological pipeline to Europe's top 5 leagues (La Liga, Serie A, Bundesliga, Ligue 1) and the UEFA Champions League.
2. **Granular Expected Goals ($xG$) Modeling**: Incorporating shot-level coordinate data, shot pressure metrics, and post-shot probability distributions into pre-match form vectors.
3. **Player Availability and Squad Rotation Modeling**: Ingesting verified starting lineups, player injury reports, and international duty fatigue factors announced 1 hour before kickoff.
4. **Deep Sequence Modeling**: Exploring recurrent neural networks (LSTMs) or temporal transformers to learn dynamic team state representations directly from sequences of match events.
5. **Advanced Calibration Algorithms**: Implementing temperature scaling or beta calibration parameterized directly on match importance and end-of-season table pressure.
6. **Containerized Cloud Deployment**: Packaging the system as a Dockerized microservice orchestrated with automated CI/CD and Redis caching on Kubernetes or AWS ECS.

---

## Academic Verification & Delivery Sign-Off

- **Final Champion Model**: Random Forest (`RandomForestClassifier`)
- **Temporal CV Mean Accuracy**: 50.39% ($\pm 1.85\%$)
- **Untouched Test Set Accuracy**: 50.66% (+7.24% lift over 43.42% naive baseline)
- **Untouched Test Set Log Loss**: 0.9964 | **Multiclass Brier Score**: 0.5925
- **53-Feature Pre-Match Schema**: Verified 100% parity across training, testing, and live feeds with zero lookahead bias.
- **Repository Status**: **FINAL STATUS: READY FOR SUBMISSION**

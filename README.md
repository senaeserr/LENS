# LENS

**LENS is a product analytics intelligence layer that turns raw event data into metrics, findings, and next decisions.**

Built as a portfolio MVP for product, growth, and data analytics workflows.

### Live demo

**Try LENS:** [lens-analytics.streamlit.app](https://lens-analytics.streamlit.app/)

---

## What LENS does

LENS helps turn raw product event data into structured analytics without requiring users to manually build every analysis from scratch.

The current MVP supports event-level SaaS / subscription product data and includes:

- Automatic schema detection
- User / event / timestamp mapping
- Event-role mapping
- Data quality checks
- Acquisition analysis
- Activation and funnel analysis
- Retention analysis
- Monetization and subscription metrics
- Payment failure monitoring
- Product and release comparisons
- Automatic findings and investigation prompts

A built-in synthetic demo dataset is included so LENS can be explored instantly without uploading a file.

---

## Why I built it

Most analytics tools assume that data is already clean, mapped, and ready for analysis.

LENS starts one step earlier.

The idea was to build a lightweight analytics layer that can understand the structure of raw event data, identify meaningful product events, calculate relevant product metrics, and surface signals worth investigating.

Instead of stopping at a dashboard, LENS is designed around the flow:

**Raw data → structure → metrics → findings → investigation → decision**

---

## Current scope

LENS is currently an **MVP focused on event-level SaaS and subscription product analytics**.

At this stage, LENS expects an event dataset containing concepts such as:

- User identifiers
- Event names
- Event timestamps
- Acquisition context
- Device / platform information
- Product or app versions
- Subscription and payment events
- Optional monetization attributes

The current version is intentionally focused on this data model so that the analytics logic can remain deterministic and interpretable.

LENS is **not yet intended to be a universal analytics engine for every data model or industry.**

---

## Demo dataset

The repository includes a synthetic SaaS dataset designed to preserve realistic product and business patterns across:

- Acquisition channels
- Devices
- Product releases
- Activation
- Retention
- Subscription conversion
- Pricing friction
- Payment failures
- Revenue
- Subscription renewals and churn

The demo contains approximately:

- **12,000 users**
- **233,000 events**
- **20 MB of event data**

Sampling was performed at the user level so complete user journeys remain intact.

The demo data is entirely synthetic and does not contain real customer information.

---

## Future work

Planned directions include:

- **Ask LENS** — natural-language investigation of product metrics and findings
- Support for **user / customer tables**
- Support for **transaction tables**
- Support for **subscription and session tables**
- Multi-table analytics across different data grains
- Domain-specific support for:
  - E-commerce
  - Mobile games
  - Marketplaces
  - Media / content products
  - Fintech products
- A/B test analysis
- Cohort comparison
- Anomaly detection
- More advanced lifecycle analysis
- Custom metric definitions

Ask LENS is planned as a language-understanding layer, while actual metric calculations remain inside LENS's deterministic analytics engine.

---

## Tech stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- RapidFuzz

---

## Project structure

```text
LENS/
├── src/
│   ├── app.py
│   ├── lens_core.py
│   ├── lens_dashboards.py
│   ├── lens_findings.py
│   ├── lens_ui.py
│   └── schema_mapper.py
│
├── data/
│   └── lens_demo_events.csv
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── .gitignore
└── README.md
```
## Run locally


Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run src/app.py
```

## Data privacy and responsible use

The included demo dataset is synthetic and contains no real customer data.
> **Do not upload personal, confidential, proprietary, regulated, or otherwise sensitive data to the public demo deployment.**

Examples include:

- Personally identifiable information
- Customer names, emails or phone numbers
- Financial account information
- Health information
- Credentials or API keys
- Confidential company datasets
- Production data containing private user information

For sensitive or proprietary datasets, LENS should be run in an appropriately secured local or private environment instead of the public portfolio deployment.

## Status

**MVP v1**

The current release focuses on:

> **Event data → semantic mapping → deterministic product metrics → automatic findings**

Future versions will expand the supported data models and introduce the Ask LENS investigation layer.

## Author

**Sena Eser**

---

## Usage rights

© 2026 Sena Eser. All rights reserved.

This repository is publicly visible for **portfolio, review and evaluation purposes**.

Commercial use, resale, redistribution, or incorporation of this project or its source code into a commercial product is not permitted without prior written permission from the author.

# MONUPED ABM - Streamlit App Guide

This repository contains an agent-based model (ABM) for building renovation behavior and a Streamlit interface in `app.py` to run simulations and visualize results on a map.

## What the app does

The Streamlit app lets you:
- set simulation parameters (initial year, simulation horizon, behavioral/random thresholds)
- configure attitude distribution (mean and standard deviation)
- run repeated simulation experiments
- inspect renovation outcomes for each year with an interactive map slider

The app uses these project data files:
- `Koper_Data/buildings_info.json`
- `Koper_Data/clusters.json`
- `Koper_Data/building_shape_filtered.geojson`

## Prerequisites

- Python 3.12+ (as defined in `pyproject.toml`)
- macOS/Linux terminal (commands below are shell-compatible)

## Install

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -e .
pip install scipy
```

Notes:
- `pip install -e .` installs dependencies listed in `pyproject.toml`.
- `scipy` is required by `app.py` (`scipy.stats.truncnorm`) and is installed explicitly.

## Install (uv users)

If you use `uv`, from the project root:

```bash
# install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create/use the project virtual environment and install dependencies from pyproject.toml
uv venv
source .venv/bin/activate
uv sync

# app.py requires scipy
uv pip install scipy
```

Notes:
- `uv sync` installs the dependencies declared in `pyproject.toml`.
- `uv pip install scipy` adds `scipy` to the environment for running `app.py`.

## Run the app

From the project root:

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`). Open it in your browser.

### Run with uv

You can also run the app directly with uv:

```bash
uv run streamlit run app.py
```

## How to use `app.py`

1. Open the form at the top of the page.
2. Set core simulation inputs:
	 - `Initial year`
	 - `Number of years to simulate`
	 - `Probability of authorities prohibiting renovation`
	 - `Behavioral threshold`
3. Adjust attitude distribution:
	 - `Mean attitude`
	 - `Attitude standard deviation`
4. Click `Run simulation`.
5. After completion:
	 - use the year slider to choose a target year
	 - inspect map colors:
		 - red: not renovated
		 - yellow: renovated in the recent period
		 - green scale: renovated by target year, colored by renovation probability

## Optional: run non-UI script

If you want a CSV output without the Streamlit UI:

```bash
python main.py
```

This writes simulation output to `outputs/simulation_results.csv`.

# Expected departure clearance times after a published airport acceptance rate

Public experiment code and processed tables for the manuscript *A liquid neural assignment of expected departure clearance times after a published airport acceptance rate*.

The Federal Aviation Administration Air Traffic Control System Command Center publishes an airport acceptance rate, then issues extra ground wait as expected departure clearance times. This repository rebuilds delay-predictive and declared landing-rate rules on Bureau of Transportation Statistics On-Time Performance files and Iowa Environmental Mesonet weather reports, converts each rate into a 15-minute extra-wait total, and assigns that total with a two-timescale liquid neural network.

## What this repository contains

- `src/` — assignment scripts used in the paper tables (`run_liquid_replace.py`, `run_casa_chen.py`, and helpers)
- `data/README.txt` — how to obtain the public monthly On-Time archives
- `results/liquid_replace/` — processed cascade tables at Atlanta, Dallas/Fort Worth, and Newark
- `results/casa_chen.json` — Computer-Assisted Slot Allocation and incremental-search comparisons with the same interval totals
- `requirements.txt`

## What this repository does not contain

Raw Bureau of Transportation Statistics monthly zip archives. Download them from TranStats and place them as described in `data/README.txt`.

- On-Time Performance: https://www.transtats.bts.gov/
- Iowa Environmental Mesonet: https://mesonet.agron.iastate.edu/request/download.phtml

## Software

Python 3.11+, `pandas`, `numpy`, `scikit-learn`, `torch`.

```text
pip install -r requirements.txt
set PYTHONPATH=src
python src/run_liquid_replace.py all
```

On Unix, replace `set PYTHONPATH=src` with `export PYTHONPATH=src`. The script expects the On-Time zip files under `data/bts/` as named in `data/README.txt`. Processed tables in `results/` can be read without rerunning the pipeline.

## License

MIT. Bureau of Transportation Statistics records and Iowa Environmental Mesonet extracts remain under their original public terms.

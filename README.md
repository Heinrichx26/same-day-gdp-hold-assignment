# Same-day assignment of extra ground hold

Public experiment code for the manuscript *Assigning extra ground hold to protect connecting turns when the airport landing rate is already set*.

The Federal Aviation Administration Air Traffic Control System Command Center sets an airport acceptance rate, then writes extra ground hold as expected departure clearance times. This repository rebuilds three 2025 landing-rate rules on Bureau of Transportation Statistics On-Time Performance files and Iowa Environmental Mesonet weather reports, converts each rate into a daily extra-hold total, and assigns that total by equal hold, 15-minute greedy assignment, and same-day assignment.

## What this repository contains

- `src/` — Python scripts used in the paper tables (`run_connect_families.py` and helpers)
- `data/README.txt` — how to obtain the public monthly On-Time archives
- `data/metar/` — Iowa Environmental Mesonet extracts used in the paper
- `results/winter/connect_families.json` — processed assignment tables used in Section 5
- `requirements.txt`

## What this repository does not contain

Raw Bureau of Transportation Statistics monthly zip archives. Download them from TranStats and place them as described in `data/README.txt`.

- On-Time Performance: https://www.transtats.bts.gov/
- Iowa Environmental Mesonet: https://mesonet.agron.iastate.edu/request/download.phtml

## Software

Python 3.11+, `pandas`, `numpy`, `scikit-learn`, `cvxpy`, `osqp`.

```text
pip install -r requirements.txt
set PYTHONPATH=src
python src/run_connect_families.py
```

On Unix, replace `set PYTHONPATH=src` with `export PYTHONPATH=src`. The script expects the On-Time zip files under `data/bts/` as named in `data/README.txt`.

## License

MIT. Bureau of Transportation Statistics records and Iowa Environmental Mesonet extracts remain under their original public terms.

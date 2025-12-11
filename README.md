# FLAST: Know Your Neighbor - Fast Static Prediction of Test Flakiness

[![CI](https://github.com/FlakinessStaticDetection/FLAST/actions/workflows/ci.yml/badge.svg)](https://github.com/FlakinessStaticDetection/FLAST/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository is a companion page for the submission "Know Your Neighbor: Fast Static Prediction of Test Flakiness".

It contains all the material required for replicating the experiments, including: the algorithm implementation, the datasets and their ground truth, and the scripts for the experiments replication.

## Requirements

- **Python 3.11+** (tested on 3.11, 3.12, 3.13)
- **Dependencies**: numpy, scipy, scikit-learn

## Quick Start

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/FlakinessStaticDetection/FLAST
cd FLAST

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Extract the dataset
tar zxvf dataset.tgz
```

### Option 2: Using Docker

```bash
# Clone the repository
git clone https://github.com/FlakinessStaticDetection/FLAST
cd FLAST

# Run all experiments
docker compose up experiments

# Or run specific experiments
docker compose up eff-eff        # RQ1 & RQ2
docker compose up compare-pinto  # RQ3
docker compose up params         # Parameter tuning
```

### Option 3: Traditional pip install

```bash
# Clone the repository
git clone https://github.com/FlakinessStaticDetection/FLAST
cd FLAST

# Install dependencies
pip install -r requirements.txt

# Extract the dataset
tar zxvf dataset.tgz
```

## Running Experiments

### RQ1 and RQ2: Effectiveness and Efficiency

```bash
python py/eff_eff.py
```

Results will be saved to `results/eff-eff.csv`.

### RQ3: Comparison with Pinto-KNN

```bash
python py/compare_pinto.py
```

Results will be saved to `results/compare-pinto.csv`.

### Parameter Tuning

```bash
python py/params.py
```

Results will be saved to multiple CSV files in `results/`.

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=py --cov-report=term-missing
```

### Code Quality

```bash
# Lint code
ruff check py/ tests/

# Format code
ruff format py/ tests/

# Type checking
mypy py/ --ignore-missing-imports
```

### Using Docker for Development

```bash
# Run tests
docker compose up test

# Run linting
docker compose up lint

# Interactive shell
docker compose run --rm shell
```

## Project Structure

```
FLAST/
├── py/                     # Core implementation and experiment scripts
│   ├── __init__.py         # Package initialization
│   ├── flast.py            # FLAST algorithm implementation
│   ├── eff_eff.py          # RQ1 & RQ2 experiments
│   ├── compare_pinto.py    # RQ3 experiments
│   └── params.py           # Parameter tuning experiments
├── tests/                  # Unit tests
│   ├── conftest.py         # Test fixtures
│   └── test_flast.py       # FLAST algorithm tests
├── dataset/                # Dataset (extracted from dataset.tgz)
├── results/                # Experiment results
├── manual-inspection/      # Manual test case analysis
├── parameters/             # Parameter tuning visualizations
├── pseudocode/             # Algorithm pseudocode
├── .github/workflows/      # CI/CD configuration
├── pyproject.toml          # Modern Python packaging
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose services
└── README.md               # This file
```

## Algorithm

The pseudocode of FLAST is available [here](pseudocode/README.md).

## API Usage

```python
from py import flast

# Load data
flaky, non_flaky = flast.get_data_points_info(Path("dataset"), "project-name")

# Vectorize
all_data = flaky + non_flaky
vectors = flast.flast_vectorization(all_data, eps=0.3)

# Classify
params = {"algorithm": "brute", "metric": "cosine", "weights": "distance"}
train_time, test_time, predictions = flast.flast_classification(
    train_data, train_labels, test_data,
    sigma=0.5, k=7, params=params
)

# Evaluate
precision, recall = flast.compute_results(test_labels, predictions)
```

## Parameter Investigation

The investigation on the effect of FLAST's parameters is available [here](parameters/README.md).

## Datasets

The project includes 13 open-source Java projects:
- achilles, alluxio-tachyon, ambari, hadoop, jackrabbit-oak
- jimfs, ninja, okhttp, oozie, oryx
- spring-boot, togglz, wro4j

Each dataset contains:
- `flakyMethods/`: Tokenized flaky test methods
- `nonFlakyMethods/`: Tokenized non-flaky test methods

## Citation

If you use this tool in your research, please cite:

```bibtex
@inproceedings{flast2021,
  title={Know Your Neighbor: Fast Static Prediction of Test Flakiness},
  author={...},
  booktitle={...},
  year={2021}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

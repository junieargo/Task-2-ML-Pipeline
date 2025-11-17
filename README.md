# Task-2-ML-Pipeline
Machine Learning Engineer - Take Home Test

bsort_project/
├── .github/
│   └── workflows/
│       └── ci_cd.yml       # GitHub Actions
├── bsort/
│   ├── __init__.py
│   ├── main.py             # CLI Entrypoint
│   ├── train.py            # Training Logic
│   ├── infer.py            # Inference Logic
│   ├── utils.py            # Helper functions (Color mapping)
│   └── preprocessing.py    # Script to fix labels based on color
├── tests/
│   ├── __init__.py
│   └── test_bsort.py       # Unit tests
├── configs/
│   └── settings.yaml       # Configuration
├── Dockerfile
├── pyproject.toml
├── README.md
└── .gitignore
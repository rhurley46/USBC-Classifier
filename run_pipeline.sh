echo "**Running Pipeline**"

python -m src.001_preprocessing --config params.yaml

python -m src.002_feature-engineering --config params.yaml

python -m src.003_train --config params.yaml


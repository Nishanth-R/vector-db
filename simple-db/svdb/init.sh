#! /bin/bash

touch bow.pickle
touch vectors.pickle

echo "Database initialized"

poetry install
poetry build

echo "Dependencies installed"

poetry run python -m build

echo "Package built"


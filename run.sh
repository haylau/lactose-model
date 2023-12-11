#!/bin/bash

required_packages=("pandas" "sklearn" "joblib" "nltk")

for package in "${required_packages[@]}"; do
    if ! pip show "$package" > /dev/null 2>&1; then
        echo "$package is not installed. Installing..."
        pip install "$package"
    fi
done

python src/train.py
python src/test.py

exit 0

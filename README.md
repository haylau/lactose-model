Last built on a Windows machine using Python 3.11.0 on 12/11/23

# Required packages:

- "pandas"
- "sklearn"
- "joblib"
- "nltk"

# To build:

Either:

- Run `run.sh`: This will install packages and run `train.py` and `test.py`
  This requires a bash environment like Git Bash or a Unix/Linux machine
  e.x.:
  `./run.sh`

- Separately install packages and run `train.py` and `test.py`
  e.x.:
  `sudo pip install pandas sklearn joblib nltk`
  `./src/train.py`
  `./src/test.py`

# Note:

A subset of the original dataset has been annotated, `lact_sample.csv`, and is what is used to train the model. `lact_testmodel.csv` is a separate dataset from the Olive Garden menu that has been annotated to be tested against after the model has been trained.

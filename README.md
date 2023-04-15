# CS598_DLH_Project

## Env Setup
Setup venv
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset preperation

Convert input Harvard Medical Data xml to csv
```shell
python3 ./dataset/preprocessing/xml_to_csv.py
```

Data preprocessing on CSV files as mentioned in the research paper
```shell
python3 prepare_preprocessed_dataset.py
```

Create Weka dataset files for CML models using tfidf representation
```shell
python3 prepare_weka_dataset_tfidf.py
```

Create Weka dataset files for CML models using tfidf representation
```shell
python3 prepare_weka_dataset_we.py
```
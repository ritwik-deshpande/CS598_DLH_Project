echo "Setting up the dataset"
python3 ./dataset/preprocessing/xml_to_csv.py
python3 prepare_preprocessed_dataset.py

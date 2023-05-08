# CS598_DLH_Project

Ensembling Classical Machine Learning and Deep Learning Approaches for Morbidity Identification From Clinical Notes

### Motivation
In our research project, we have examined the paper titled 'Ensembling Classical Machine Learning and Deep Learning Approaches for Morbidity Identification From Clinical Notes' \cite{tp1}. The paper titled 'Ensembling Classical Machine Learning and Deep Learning Approaches for Morbidity Identification From Clinical Notes' addresses the challenge of identifying multiple morbidity factors in patients using their clinical records. 

The authors suggest using classical machine learning and deep learning approaches to automatically detect morbidity factors, which can assist healthcare personnel in handling massive volumes of electronic health records. The paper explores different feature representation methods such as word embeddings and bag-of-words, in combination with CML and DL approaches. Furthermore, it investigates the use of ensemble methods to enhance classification accuracy. The paper concludes that ensemble strategies show better performance for morbidity identification and can effectively leverage both CML and DL approaches.


### Env Setup
Setup venv
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset preperation

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

### References
Vivek Kumar, Diego Reforgiato Recupero, Daniele Riboni, and Rim Helaoui. 2021. Ensembling classical machine learning and deep learning approaches for morbidity identification from clinical notes. IEEE Access, 9:7107–7126.

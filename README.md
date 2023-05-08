# CS598_DLH_Project

Ensembling Classical Machine Learning and Deep Learning Approaches for Morbidity Identification From Clinical Notes

### Motivation
In our research project, we have examined the paper titled 'Ensembling Classical Machine Learning and Deep Learning Approaches for Morbidity Identification From Clinical Notes' \cite{tp1}. The paper titled 'Ensembling Classical Machine Learning and Deep Learning Approaches for Morbidity Identification From Clinical Notes' addresses the challenge of identifying multiple morbidity factors in patients using their clinical records. 

The authors suggest using classical machine learning and deep learning approaches to automatically detect morbidity factors, which can assist healthcare personnel in handling massive volumes of electronic health records. The paper explores different feature representation methods such as word embeddings and bag-of-words, in combination with CML and DL approaches. Furthermore, it investigates the use of ensemble methods to enhance classification accuracy. The paper concludes that ensemble strategies show better performance for morbidity identification and can effectively leverage both CML and DL approaches.


### Dependencies
All the dependencies can be found in the following two files:  
[requirements.txt](https://github.com/ritwik-deshpande/CS598_DLH_Project/blob/main/requirements.txt)   
[dl_requirements.txt](https://github.com/ritwik-deshpande/CS598_DLH_Project/blob/main/dl_requirements.txt)


### Data Download Instructions
We have used the n2c2 NLP Research Data Sets(Unstructured notes from the Research Patient Data Registry at Partners Healthcare (originally developed during the i2b2 project)). For this project, we requested access to the above dataset on the [Harvard Medical School Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).


### Env Setup
Setup venv
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r dl_requirements.txt
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

Deep Learning
```shell
cd models/DL/word-embeddings
python3 bilstm_funcx.py
```

### Preprocessing
All the files for preprocessing the data and generating feature representations can be found here:  
[Preprocessing](https://github.com/ritwik-deshpande/CS598_DLH_Project/tree/main/dataset/preprocessing)

### Models
All the files in which we train, test and evaluate the CML, DL and ensemble models can be found here:  
[Models](https://github.com/ritwik-deshpande/CS598_DLH_Project/tree/main/models)

### Implementation

#### CML
For CML models we have leveraged the scikit learn library which are trained on an Apple Macbook Air M1 8GB PC. For JRip and J48 we have leveraged the weka3.7.8.jar file and utilized the python-weka-wrapper3 and java-bridge to call the respective processes running on the JVM through our python SDK. 

#### Deep Learning

For Deep Learning models we have used the pytorch framework that trains the BiLSTM models on Nvidia A100 GPUs on High Performance Clusters at NCSA, using a fire and forget model provided by globus-compute SDK.

[Function-as-a-service (globus-compute) ](https://www.globus.org/compute)


### Results
All the files containing micro F1 and macro F1 scores can be found here:  
[Results](https://github.com/ritwik-deshpande/CS598_DLH_Project/tree/main/results)

#### 1. Effect of Feature Selection  

Performance of Random Forest with SelectKBest for TF-IDF features  

|Morbidity Class     |RF_Macro F1       |RF_Micro F1       |
|--------------------|------------------|------------------|
|Asthma              |0.9919438722280273|0.992029702970297 |
|CAD                 |0.9328710844519182|0.9338461538461539|
|CHF                 |1                 |1                 |
|Depression          |0.9339119948939153|0.9347826086956521|
|Diabetes            |0.9695515980536025|0.9697310126582279|
|Gallstones          |0.924511886671068 |0.9258687633469231|
|GERD                |0.8716512823340518|0.8736036036036035|
|Gout                |0.948954179021316 |0.9497386109036594|
|Hypercholesterolemia|0.8784615738633811|0.8814586357039188|
|Hypertension        |0.96933086719561  |0.9696716826265389|
|Hypertriglyceridemia|0.9817566408751469|0.9819492219492221|
|OA                  |0.9432050987861824|0.9442690459849004|
|Obesity             |0.97722111138428  |0.9777009728622634|
|OSA                 |0.9756506637702633|0.9762764511745292|
|PVD                 |0.9704446989641516|0.97124227865477  |
|Venous-Insufficiency|0.9737180229679897|0.9741194158075601|
|Overall-Average     |0.9526990359663066|0.9535180100492637|

Performance of Random Forest with All features for TF-IDF features  

|Morbidity Class     |RF_Macro F1       |RF_Micro F1       |
|--------------------|------------------|------------------|
|Asthma              |0.9829017417388215|0.9830693069306932|
|CAD                 |0.9329209174682859|0.9338461538461539|
|CHF                 |1                 |1                 |
|Depression          |0.933505289404085 |0.9347826086956521|
|Diabetes            |0.9656652065048952|0.9659018987341772|
|Gallstones          |0.9416357626126451|0.9426616191030869|
|GERD                |0.8892573399211786|0.891081081081081 |
|Gout                |0.9566426264804031|0.9575149365197909|
|Hypercholesterolemia|0.8245578746271189|0.8281567489114661|
|Hypertension        |0.9504351152296595|0.9510396716826266|
|Hypertriglyceridemia|0.9845085956132467|0.9846601146601147|
|OA                  |0.9421724035016409|0.9431937771676961|
|Obesity             |0.9238661441337006|0.9251920122887866|
|OSA                 |0.9776611448543944|0.9782566491943312|
|PVD                 |0.9670155760085459|0.9680507892930679|
|Venous-Insufficiency|0.974672820924463 |0.9751396048109967|
|Overall-Average     |0.9467136599389426|0.9476591858074825|  

Performance improves with feature selection.

#### 2. Performance of deep learning classifiers on the given dataset using different word embeddings.

|Embeddings          |Avg Macro F1      |Avg Micro F1      |
|--------------------|------------------|------------------|
|Word2Vec            |65.59             |82.06             |
|Glove               |70.41             |84.26             |
|FastText            |65.43             |82.03             |
|USE                 |71.77             |85.37             |  

GloVe performed the best with BiLSTM DL model.

#### 3. Avg F1 scores for BiLSTM and LSTM DL models

|Models|Avg Macro F1|Avg Micro F1|
|------|------------|------------|
|BiLSTM|0.683       |0.834       |
|LSTM  |0.664       |0.823       |  

BiLSTM performed better than LSTM DL model.


### References
Vivek Kumar, Diego Reforgiato Recupero, Daniele Riboni, and Rim Helaoui. 2021. Ensembling classical machine learning and deep learning approaches for morbidity identification from clinical notes. IEEE Access, 9:7107–7126.

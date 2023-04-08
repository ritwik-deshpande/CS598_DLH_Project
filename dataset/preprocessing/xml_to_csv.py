import csv
import xmltodict
def convert():
    with open('../train/obesity_patient_records_training.xml') as fd:
        xml_data_train = fd.read()
        xml_dict_train = xmltodict.parse(xml_data_train)

    # Print the dictionary

    with open('../train/obesity_standoff_intuitive_annotations_training.xml') as fd:
        xml_data_train_annotation_intuitive = fd.read()
        xml_dict_train_annotation_intuitive = xmltodict.parse(xml_data_train_annotation_intuitive)

    with open('../train/obesity_standoff_textual_annotations_training.xml') as fd:
        xml_data_train_annotation_textual = fd.read()
        xml_dict_train_annotation_textual = xmltodict.parse(xml_data_train_annotation_textual)

    diseases = []
    train_data_intuitive_dict = {}
    train_data_textual_dict = {}

    for xml_item in xml_dict_train['root']['docs']['doc']:
        train_data_intuitive_dict[int(xml_item['@id'])] = {'text': xml_item['text']}
        train_data_textual_dict[int(xml_item['@id'])] = {'text': "This is a sample Text with jumps, jumped jumping RANdom. . punctuations and \ '. '"}

    for xml_item in xml_dict_train_annotation_intuitive['diseaseset']['diseases']['disease']:
        disease_name = xml_item['@name'].strip().replace(' ','-')
        diseases.append(disease_name)
        for doc in xml_item['doc']:
            doc_id = int(doc['@id'])
            judgement = doc['@judgment']
            train_data_intuitive_dict[doc_id][disease_name] = judgement

    for xml_item in xml_dict_train_annotation_textual['diseaseset']['diseases']['disease']:
        disease_name = xml_item['@name'].strip().replace(' ','-')

        for doc in xml_item['doc']:
            doc_id = int(doc['@id'])
            judgement = doc['@judgment']
            train_data_textual_dict[doc_id][disease_name] = judgement


    with open('../train/train_data_intuitive.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        cols = ['Doc_id', 'text'] + diseases
        # print(cols)
        writer.writerow(['Doc_id', 'text'] + diseases)
        for doc_id, values in train_data_intuitive_dict.items():
            row = []
            row.append(doc_id)
            for col in cols[1:]:
                if col not in values:
                    row.append('UNK')
                else:
                    row.append(values[col])

            writer.writerow(row)

    with open('../train/train_data_textual.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        cols = ['Doc_id', 'text'] + diseases
        # print(cols)
        writer.writerow(['Doc_id', 'text'] + diseases)
        for doc_id, values in train_data_textual_dict.items():
            row = []
            row.append(doc_id)
            for col in cols[1:]:
                if col not in values:
                    row.append('UNK')
                else:
                    row.append(values[col])

            writer.writerow(row)





if __name__ =='__main__':
    convert()
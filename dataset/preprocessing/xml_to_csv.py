import csv
import xmltodict
import os

def write_to_csv(output_file_name, data_dict, diseases):
    with open(output_file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        cols = ['Doc_id', 'text'] + diseases
        writer.writerow(['Doc_id', 'text'] + diseases)
        for doc_id, values in data_dict.items():
            row = []
            row.append(doc_id)
            for col in cols[1:]:
                if col not in values:
                    row.append('UNK')
                else:
                    row.append(values[col])
            writer.writerow(row)

def convert(inp_train_text_xml_file, inp_train_intuitive_xml, inp_train_textual_xml, out_intuitive_csv_file,
            out_textual_csv_file):
    with open(inp_train_text_xml_file) as fd:
        xml_data = fd.read()
        xml_dict = xmltodict.parse(xml_data)

    with open(inp_train_intuitive_xml) as fd:
        xml_data_annotation_intuitive = fd.read()
        xml_dict_annotation_intuitive = xmltodict.parse(xml_data_annotation_intuitive)

    with open(inp_train_textual_xml) as fd:
        xml_data_annotation_textual = fd.read()
        xml_dict_annotation_textual = xmltodict.parse(xml_data_annotation_textual)

    diseases = []
    data_intuitive_dict = {}
    data_textual_dict = {}

    for xml_item in xml_dict['root']['docs']['doc']:
        data_intuitive_dict[int(xml_item['@id'])] = {'text': xml_item['text']}
        data_textual_dict[int(xml_item['@id'])] = {'text': xml_item['text']}

    for xml_item in xml_dict_annotation_intuitive['diseaseset']['diseases']['disease']:
        disease_name = xml_item['@name'].strip().replace(' ', '-')
        diseases.append(disease_name)
        for doc in xml_item['doc']:
            doc_id = int(doc['@id'])
            judgement = doc['@judgment']
            data_intuitive_dict[doc_id][disease_name] = judgement

    for xml_item in xml_dict_annotation_textual['diseaseset']['diseases']['disease']:
        disease_name = xml_item['@name'].strip().replace(' ', '-')
        for doc in xml_item['doc']:
            doc_id = int(doc['@id'])
            judgement = doc['@judgment']
            data_textual_dict[doc_id][disease_name] = judgement

    write_to_csv(out_intuitive_csv_file, data_intuitive_dict, diseases)
    write_to_csv(out_textual_csv_file, data_textual_dict, diseases)


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    convert('../train/obesity_patient_records_training.xml',
            '../train/obesity_standoff_intuitive_annotations_training.xml',
            '../train/obesity_standoff_textual_annotations_training.xml',
            '../train/train_data_intuitive.csv',
            '../train/train_data_textual.csv')

    print("Converted train xml to csv")

    convert('../test/obesity_patient_records_test.xml',
            '../test/obesity_standoff_annotations_test_intuitive.xml',
            '../test/obesity_standoff_annotations_test_textual.xml',
            '../test/test_data_intuitive.csv',
            '../test/test_data_textual.csv')

    print("Converted test xml to csv")

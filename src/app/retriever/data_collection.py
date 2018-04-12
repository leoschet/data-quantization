from enum import Enum
import numpy as np
import codecs
import arff
import os

class EDataType(Enum):
    """
    Enumerable class with all data types
    """
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'
    MIXED = 'mixed'

class DataCollection(object):
    def __init__(self, corpus_dir):
        self._corpus_dir = corpus_dir
        self.documents = self._get_documents()

    def get_data_label(self, relation):
        relation_data = self.documents[relation]['data']

        labeled_data = [(data[:(len(data) - 1)], data[(len(data) - 1)]) 
                        for index, data in enumerate(relation_data)]

        data = [data for data, _ in labeled_data]
        labels = [label for _, label in labeled_data]

        return np.array(data), np.array(labels)

    def get_features_types(self, relation):
        relation_attr = self.documents[relation]['attributes']
        relation_attr = relation_attr[:(len(relation_attr) - 1)]
        return [EDataType.CATEGORICAL if isinstance(attr_type, list)
                else EDataType.NUMERICAL for _, attr_type in relation_attr]

    def _get_documents(self):
        files_name = os.listdir(self._corpus_dir)
        file_relation = [self._get_document(self._corpus_dir + file_name) for file_name in files_name
                 if file_name.endswith(".arff")]

        files = {}
        for file_, relation in file_relation:
            assert relation not in files
            files[relation] = file_

        return files

    def _get_document(self, path):
        file_ = codecs.open(path, 'rb', 'utf-8')
        data = arff.load(file_)
        return (data, data['relation'])

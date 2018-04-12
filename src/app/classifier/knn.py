from ..retriever.data_collection import EDataType
import numpy as np

class KNearestNeighbors(object):

    @property
    def train_data(self):
        """
        Read-only property
        """
        return self._train_data

    @property
    def data_type(self):
        """
        Read-only property
        """
        return self._data_type

    def __init__(self, k, weighted_distance):
        self.k = k
        self.weighted_distance = weighted_distance

    def fit(self, train_data, train_labels, features_types):
        assert (EDataType.NUMERICAL in features_types or
                EDataType.CATEGORICAL in features_types)

        if EDataType.NUMERICAL in features_types:
            self._data_type = EDataType.NUMERICAL
            if EDataType.CATEGORICAL in features_types:
                self._data_type = EDataType.MIXED
        else:
            self._data_type = EDataType.CATEGORICAL
        
        self._errors_index = []
        self._train_data = train_data
        self._train_labels = train_labels
        self._data_labels = set(train_labels)
        self._prob_dict = self._calculate_data_probabilities()

        self._numerical_features_indexes = [index for index, attr_type in enumerate(features_types)
                                            if attr_type is EDataType.NUMERICAL]
        self._categorical_features_indexes = [index for index, attr_type in enumerate(features_types)
                                              if attr_type is EDataType.CATEGORICAL]

        self._standard_deviations = self._calculate_numerical_features_std()

    def _calculate_data_probabilities(self):
        feature_info = []
        for d_index, data in enumerate(self._train_data):
            if not feature_info:
                feature_info = [{}] * len(data)

            d_label = self._train_labels[d_index]
            for f_index, f_value in enumerate(data):
                if f_value not in feature_info[f_index]:
                    feature_info[f_index][f_value] = {}
                if d_label not in feature_info[f_index][f_value]:
                    feature_info[f_index][f_value][d_label] = []
                feature_info[f_index][f_value][d_label].append(d_index)

        prob_dict = {}
        for f_index, f_dict in enumerate(feature_info):
            for f_value in f_dict.keys():
                f_value_count = 0
                f_value_class_count = {}
                for d_label in feature_info[f_index][f_value]:
                    class_count = len(feature_info[f_index][f_value][d_label])
                    f_value_class_count[d_label] = class_count
                    f_value_count += class_count

                if f_index not in prob_dict:
                    prob_dict[f_index] = {}
                if f_value not in prob_dict[f_index]:
                    # (f_value in f_index count, class_prob dictionary)
                    prob_dict[f_index][f_value] = {}

                for d_label in f_value_class_count:
                    class_count = f_value_class_count[d_label]
                    prob_dict[f_index][f_value][d_label] = class_count/f_value_count
        
        return prob_dict
    
    def _calculate_numerical_features_std(self):
        standard_deviations = {}
        for index in self._numerical_features_indexes:
            try:
                std = np.std([float(value) for value in self._prob_dict[index].keys()])
            except:
                self._errors_index.append(index)
                continue
            standard_deviations[index] = std
        
        return standard_deviations

    def _get_nearest(self, in_features):
        """
        Returns an array of tuples (distance, label) with the k-nearest neighbors from in_features.
        """
        labeled_distances = []
        for index, features in enumerate(self._train_data):
            distance = self._calculate_distance(in_features, features)
            labeled_distances.append((distance, self._train_labels[index]))
        
        labeled_distances.sort(key=lambda tup: tup[0], reverse=self.weighted_distance)
        return labeled_distances[:self.k]

    def _calculate_distance(self, in_features, features):
        distance = -1

        if self._data_type == EDataType.NUMERICAL:
            distance = self._euclidean_distance(in_features, features)
        elif self._data_type == EDataType.CATEGORICAL:
            distance = self._vdm_distance(in_features, features)
        elif self._data_type == EDataType.MIXED:
            distance = self._hvdm_distance(in_features, features)

        if (self.weighted_distance):
            if distance == 0:
                distance = 1

            distance = 1/(distance ** 2)

        # TODO: Check better way to treat error case
        return distance

    def _euclidean_distance(self, in_features, features):
        assert len(in_features) == len(features)

        distance = 0
        for f_index, f_value in enumerate(features):
            in_value = in_features[f_index]
            distance +=  self._local_euclidean_distance(f_index, f_value, in_value)

        return distance ** (1/2)
    
    def _local_euclidean_distance(self, f_index, f1, f2):
        # TODO: Add pre condition
        numerator = (float(f1) - float(f2)) ** 2
        denominator = 4 * self._standard_deviations[f_index] # Data normalization
        return  numerator/denominator

    def _vdm_distance(self, in_features, features):
        assert len(in_features) == len(features)

        distance = 0
        for f_index, f_value in enumerate(features):
            in_value = in_features[f_index]
            distance += self._local_vdm_distance(f_index, f_value, in_value)

        return distance ** (1/2) # Data normalization

    def _local_vdm_distance(self, f_index, f1, f2):
        local_vdm = 0

        pre_cond = (f_index in self._prob_dict and
                    f1 in self._prob_dict[f_index] and
                    f2 in self._prob_dict[f_index])

        if pre_cond:
            q = 2 # Usually 1 or 2        
            for d_label in self._data_labels:
                prob1 = 0
                if d_label in self._prob_dict[f_index][f1]:
                    prob1 = self._prob_dict[f_index][f1][d_label]

                prob2 = 0
                if d_label in self._prob_dict[f_index][f2]:
                    prob2 = self._prob_dict[f_index][f2][d_label]

                local_vdm += abs(prob1 - prob2) ** q

        return local_vdm

    def _hvdm_distance(self, in_features, features):
        assert len(in_features) == len(features)
        
        distance = 0
        for f_index in self._numerical_features_indexes:
            if f_index in self._errors_index:
                continue
            in_value = in_features[f_index]
            f_value = features[f_index]
            distance += self._local_euclidean_distance(f_index, f_value, in_value) ** 2

        for f_index in self._categorical_features_indexes:
            if f_index in self._errors_index:
                continue
            in_value = in_features[f_index]
            f_value = features[f_index]
            # It is not necessary to square the local distance,
            # since the vdm_distance normalization would take square root it square root
            distance += self._local_vdm_distance(f_index, f_value, in_value)

        return distance ** (1/2)

class KNeighborsClassifier(KNearestNeighbors):
    def __init__(self, k, weighted_distance=True):
        super(KNeighborsClassifier, self).__init__(k, weighted_distance)

    def predict(self, in_features):
        k_nearest = self._get_nearest(in_features) # They come as a tuple: (distance, label)

        possible_labels = {}
        for distance, label in k_nearest:
            if label not in possible_labels:
                possible_labels[label] = []

            possible_labels[label].append(distance)
        
        ordered_label_distances = list(possible_labels.items())
        if (self.weighted_distance):
            ordered_label_distances.sort(key=lambda tup: sum(tup[1]), reverse=True)
        else:
            ordered_label_distances.sort(key=lambda tup: len(tup[1]), reverse=True)
        
        return ordered_label_distances

class KNeighborsRegressor(KNearestNeighbors):
    def __init__(self, k, weighted_distance=True):
        super(KNeighborsRegressor, self).__init__(k, weighted_distance)

    def predict(self, in_features):
        k_nearest = self._get_nearest(in_features) # They come as a tuple: (distance, value)

        prediction = 0
        for _, value in k_nearest:
            prediction += (value/self.k)

        return prediction

from ..retriever.data_collection import EDataType
from ..classifier.knn import KNearestNeighbors
import numpy as np

class LinearVectorQuantization(KNearestNeighbors):
    @property
    def prototypes(self):
        """
        Read-only property
        """
        return np.copy(self._prototypes)
    
    @property
    def prototypes_labels(self):
        """
        Read-only property
        """
        return np.copy(self._prototypes_labels)
    
    def __init__(self, prototype_selection, prototypes_per_class=1, weighted_distance=True):
        super(LinearVectorQuantization, self).__init__(1, weighted_distance)
        self.prototypes_per_class = prototypes_per_class
        self.prototype_selection = prototype_selection

    def fit(self, train_data, train_labels, features_types):
        super(LinearVectorQuantization, self).fit(train_data, train_labels, features_types)
        # Get initial prototypes indexes
        prototypes_indexes = self.prototype_selection(self._train_data, self._train_labels, self.prototypes_per_class)
        # Get initial prototypes data
        self._prototypes = self._train_data[prototypes_indexes]
        # Get initial prototypes labels
        self._prototypes_labels = self._train_labels[prototypes_indexes]

    def fit_prototypes(self, learning_rate, verbose=0):
        if verbose > 0:
            print('Prototypes_data:', self._prototypes)
            print('Prototypes_labels:', self._prototypes_labels)

        for index, data_features in enumerate(self._train_data):
            # Get data_features' nearest prototypes indexes
            nearest_index = self._get_nearest_prototypes(data_features, only_indexes=True)
            nearest_index = nearest_index[0]
            # Extract only the labels
            nearest_label = self._prototypes_labels[nearest_index]

            # Get prototypes_labels mode
            label_mode = self.mode(nearest_label)
            # Get current data's label
            data_label = self._train_labels[index]

            prototype = self._prototypes[nearest_index]
            if data_label == label_mode:
                prototype = self._adjust_prototype(prototype, data_features, learning_rate, approximate=True)
            else:
                prototype = self._adjust_prototype(prototype, data_features, learning_rate, approximate=False)

            self._prototypes[nearest_index] = prototype

            if verbose > 0:
                print_rate = int((len(self._train_data)/4)/verbose)
                if index%print_rate == 0:
                    print('\nIteration number:', index)
                    print('\tPrototypes_data:', self._prototypes)
                    print('\tPrototypes_labels:', self._prototypes_labels)


    def mode(self, array):
        values, counts = np.unique(array, return_counts=True)
        m = counts.argmax()
        return values[m]

    def _get_nearest_prototypes(self, in_features, only_indexes=False):
        """
        Returns an array of tuples (distance, label) with the k-nearest
        neighbors of in_features from prototypes.
        """
        distance_index = []
        for index, features in enumerate(self._prototypes):
            distance = self._calculate_distance(in_features, features)
            distance_index.append((distance, index))

        distance_index.sort(key=lambda tup: tup[0], reverse=self.weighted_distance)
        distance_index = distance_index[:self.k]

        if only_indexes:
            return [index for _, index in distance_index]
        return distance_index

    def _adjust_prototype(self, prototype, data_features, learning_rate, approximate):
        adjustment = learning_rate * (prototype - data_features)
        if not approximate:
            adjustment *= -1
        return prototype + adjustment

class LinearVectorQuantization2(LinearVectorQuantization):
    def __init__(self, prototype_selection, relative_width, prototypes_per_class=1, weighted_distance=True):
        super(LinearVectorQuantization2, self).__init__(prototype_selection, prototypes_per_class, weighted_distance)
        self.k = 2
        self.window = (1-relative_width)/(1+relative_width)

    def fit_prototypes(self, learning_rate, verbose=0):
        if verbose > 0:
            print('Prototypes_data:', self._prototypes)
            print('Prototypes_labels:', self._prototypes_labels)

        for index, data_features in enumerate(self._train_data):
            # Get data_features' nearest prototypes indexes
            nearest_dist_index = self._get_nearest_prototypes(data_features, only_indexes=False)
            # Extract only the distances
            nearest_dist = [dist for dist, _ in nearest_dist_index]
            # Extract only the indexes
            nearest_index = [index for _, index in nearest_dist_index]
            # Extract only the labels
            nearest_label = self._prototypes_labels[nearest_index]
            # Get current data's label
            data_label = self._train_labels[index]
            
            dist0 = nearest_dist[0]/nearest_dist[1]
            dist1 = nearest_dist[1]/nearest_dist[0]
            dist = min(dist0, dist1)

            if dist > self.window:
                if nearest_label[0] != nearest_label[1]:
                    if data_label == nearest_label[0]:
                        self._prototypes[nearest_index[0]] = self._adjust_prototype(self._prototypes[nearest_index[0]], data_features, learning_rate, approximate=True)
                        self._prototypes[nearest_index[1]] = self._adjust_prototype(self._prototypes[nearest_index[1]], data_features, learning_rate, approximate=False)
                    else:
                        self._prototypes[nearest_index[0]] = self._adjust_prototype(self._prototypes[nearest_index[0]], data_features, learning_rate, approximate=False)
                        self._prototypes[nearest_index[1]] = self._adjust_prototype(self._prototypes[nearest_index[1]], data_features, learning_rate, approximate=True)

            if verbose > 0:
                print_rate = int((len(self._train_data)/4)/verbose)
                if index%print_rate == 0:
                    print('\nIteration number:', index)
                    print('\tPrototypes_data:', self._prototypes)
                    print('\tPrototypes_labels:', self._prototypes_labels)

class LinearVectorQuantization3(LinearVectorQuantization):
    def __init__(self, prototype_selection, relative_width, prototypes_per_class=1, weighted_distance=True):
        super(LinearVectorQuantization3, self).__init__(prototype_selection, prototypes_per_class, weighted_distance)
        self.k = 2
        self.window = (1-relative_width)/(1+relative_width)

    def fit_prototypes(self, learning_rate, verbose=0):
        if verbose > 0:
            print('Prototypes_data:', self._prototypes)
            print('Prototypes_labels:', self._prototypes_labels)

        for index, data_features in enumerate(self._train_data):
            # Get data_features' nearest prototypes indexes
            nearest_dist_index = self._get_nearest_prototypes(data_features, only_indexes=False)
            # Extract only the distances
            nearest_dist = [dist for dist, _ in nearest_dist_index]
            # Extract only the indexes
            nearest_index = [index for _, index in nearest_dist_index]
            # Extract only the labels
            nearest_label = self._prototypes_labels[nearest_index]
            # Get current data's label
            data_label = self._train_labels[index]
            
            dist0 = nearest_dist[0]/nearest_dist[1]
            dist1 = nearest_dist[1]/nearest_dist[0]
            dist = min(dist0, dist1)

            if dist > self.window:
                if nearest_label[0] != nearest_label[1]:
                    if data_label == nearest_label[0]:
                        self._prototypes[nearest_index[0]] = self._adjust_prototype(self._prototypes[nearest_index[0]], data_features, learning_rate, approximate=True)
                        self._prototypes[nearest_index[1]] = self._adjust_prototype(self._prototypes[nearest_index[1]], data_features, learning_rate, approximate=False)
                    else:
                        self._prototypes[nearest_index[0]] = self._adjust_prototype(self._prototypes[nearest_index[0]], data_features, learning_rate, approximate=False)
                        self._prototypes[nearest_index[1]] = self._adjust_prototype(self._prototypes[nearest_index[1]], data_features, learning_rate, approximate=True)
                else:
                    if data_label == nearest_label[0]:
                        self._prototypes[nearest_index[0]] = self._adjust_prototype(self._prototypes[nearest_index[0]], data_features, learning_rate, approximate=True)
                        self._prototypes[nearest_index[1]] = self._adjust_prototype(self._prototypes[nearest_index[1]], data_features, learning_rate, approximate=True)

            if verbose > 0:
                print_rate = int((len(self._train_data)/4)/verbose)
                if index%print_rate == 0:
                    print('\nIteration number:', index)
                    print('\tPrototypes_data:', self._prototypes)
                    print('\tPrototypes_labels:', self._prototypes_labels)

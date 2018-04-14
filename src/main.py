from app.retriever.data_collection import DataCollection, EDataType
from app.quantization.lvq import LinearVectorQuantization, LinearVectorQuantization2, LinearVectorQuantization3
from app.quantization import prototype_selection as ps

from knn_main import run_knn
from plot import plot_stuff

print('Collecting data...')
data_collection = DataCollection('../res/promise/')

lvqs = [None] * 3
lvqs[0] = LinearVectorQuantization(ps.first_match, prototypes_per_class=3)
lvqs[1] = LinearVectorQuantization2(ps.first_match, relative_width=0.6, prototypes_per_class=3)
lvqs[2] = LinearVectorQuantization3(ps.first_match, relative_width=0.6, prototypes_per_class=3)

lvq_results = []
ks = []
for index, lvq in enumerate(lvqs):
    lvq_data = {}
    for relation in data_collection.documents:
        lvq_data[relation] = {}
        print('\nTesting for ' + relation + ' data set')
        relation_data, relation_labels = data_collection.get_data_label(relation)
        data_len = len(relation_data)
        print('\tTotal data collected: ' + str(data_len))

        features_types = data_collection.get_features_types(relation)

        # prototypes_indexes = ps.first_match(relation_data, relation_labels)
        # print('prototypes_data', relation_data[prototypes_indexes])
        # print('prototypes_labels', relation_labels[prototypes_indexes])

        lvq.fit(relation_data, relation_labels, features_types)
        
        initial_prototypes = lvq.prototypes
        lvq.fit_prototypes(learning_rate=0.3, verbose=0)
        final_prototypes = lvq.prototypes

        lvq_data[relation]['data'] = lvq.prototypes
        lvq_data[relation]['labels'] = lvq.prototypes_labels
        lvq_data[relation]['features_types'] = features_types
        lvq_data[relation]['test_data'] = relation_data
        lvq_data[relation]['test_labels'] = relation_labels

        # print('\ninitial_prototypes:', initial_prototypes)
        # print('\nfinal_prototypes:', final_prototypes)

        # print('\n\nInitial and final prototypes euclidean distance:')
        # for index, _ in enumerate(initial_prototypes):
        #     initial = initial_prototypes[index]
        #     final = final_prototypes[index]
            
        #     distance = lvq._euclidean_distance(initial, final)
        #     print('\tPrototypes on index', index, 'has distance:', distance)
    
    lvq_result, ks = run_knn(lvq_data, str(index))
    print('\n\nlvq at index:', index, lvq_result)
    lvq_results.append(lvq_result)

for index, result in enumerate(lvq_results):
    plot_stuff(result, index+1, ks)
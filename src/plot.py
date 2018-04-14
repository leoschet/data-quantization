import matplotlib.pyplot as plt

print('Starting')
def plot_stuff(results, lvq_id, x_axis):
    for dataset in results:
        precision = results[dataset]['precision']
        recall = results[dataset]['recall']
        f1 = results[dataset]['f1']
        
        plt.figure(1)

        # Precision
        plt.subplot(311)
        plt.plot(x_axis, precision, 'k', x_axis, precision, 'r^')
        plt.yscale('linear', linthreshy=1)
        plt.title('lvq' + str(lvq_id) + ' prototypes on ' + dataset)
        plt.xlabel('K')
        plt.ylabel('Precision mean')
        plt.grid(True)

        # Recall
        plt.subplot(312)
        plt.plot(x_axis, recall, 'k', x_axis, recall, 'ro')
        plt.yscale('linear', linthreshy=1)
        plt.xlabel('K')
        plt.ylabel('Recall mean')
        plt.grid(True)

        # F1
        plt.subplot(313)
        plt.plot(x_axis, f1, 'k', x_axis, f1, 'rs')
        plt.yscale('linear', linthreshy=1)
        plt.xlabel('K')
        plt.ylabel('F1 mean')
        plt.grid(True)

        # plt.axis([1, 5, 0, 1])
        print('Ploting')
        plt.show()
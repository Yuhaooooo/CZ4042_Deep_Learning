import pylab as plt
import multiprocessing as mp
from train import train, epochs


def main():
    params = []
    param1 = {}
    param1['weight_decay_parameter'] = 1e-6
    param1['batch_size'] = 4
    param1['hidden_layer_neuron_num'] = 10
    param1['required'] = 'cross-validation accuracy'
    param1['hidden_layer_num'] = 1
    param2 = param1.copy()
    param3 = param1.copy()
    param4 = param1.copy()
    param5 = param1.copy()

    params.append(param1)
    param2['batch_size'] = 8
    params.append(param2)
    param3['batch_size'] = 16
    params.append(param3)
    param4['batch_size'] = 32
    params.append(param4)
    param5['batch_size'] = 64
    params.append(param5)
    no_threads = mp.cpu_count()
    p = mp.Pool(processes=no_threads)
    accs = p.map(train, params)
    plt.figure()
    for i in range(len(params)):
        plt.plot(range(epochs), accs[i][0], label='Batch size:'
                                                  + str(params[i]['batch_size']))
    plt.xlabel('Epochs')
    plt.ylabel('Cross Validation Accuracy')
    plt.legend()
    plt.savefig('Q2ab.png', bbox_inches='tight', dpi=100)
    plt.show()

    # plot time taken to run one epoch
    plt.figure()
    for i in range(len(params)):
        plt.scatter(params[i]['batch_size'], accs[i][1])
    plt.xlabel('Batch Size')
    plt.ylabel('Time Taken to Run One Epoch')
    plt.savefig('Q2ab_Time.png', bbox_inches='tight', dpi=100)
    plt.show()

    # plot for (c)
    param2['required'] = 'train accuracy and test accuracy'
    acc = train(param2)
    plt.figure()
    plt.plot(range(epochs), acc[0], label='Train Accuracy')
    plt.plot(range(epochs), acc[1], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Q2c.png', bbox_inches='tight', dpi=100)
    plt.show()


if __name__ == '__main__':
    main()
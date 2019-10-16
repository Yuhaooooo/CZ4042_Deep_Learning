import pylab as plt
import multiprocessing as mp
from train import train, epochs, get_data


def main():
    data = get_data()
    params = []
    param = {}
    param['weight_decay_parameter'] = 1e-6
    param['batch_size'] = 32
    param['hidden_layer_neuron_num'] = 10
    param['required'] = 'train accuracy and test accuracy'
    param['hidden_layer_num'] = 2
    param['data'] = data
    params.append(param)
    
    accs = []
    for param in params:
        acc = train(param)
        accs.append(acc)
        
    plt.figure()
    for i in range(len(params)):
        plt.plot(range(epochs), accs[i][0], label='Train Accuracy')
        plt.plot(range(epochs), accs[i][1], label='Test Accuracy')
        
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Q5.png', bbox_inches='tight', dpi=100)
    plt.show()


if __name__ == '__main__':
    main()


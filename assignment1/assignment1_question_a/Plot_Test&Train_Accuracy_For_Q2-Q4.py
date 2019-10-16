import pylab as plt
import multiprocessing as mp
from train import train, epochs, get_data


def main():
    data = get_data()
    params = []
    param1 = {}
    param1['weight_decay_parameter'] = 1e-6
    param1['batch_size'] = 64
    param1['hidden_layer_neuron_num'] = 10
    param1['required'] = 'train accuracy and test accuracy'
    param1['hidden_layer_num'] = 1
    param1['data'] = data
    params.append(param1)
    
    param2 = param1.copy()
    param2['hidden_layer_neuron_num'] = 25
    params.append(param2)
    
    param3 = param2.copy()
    param3['weight_decay_parameter'] = 0
    params.append(param3)
    
    no_threads = mp.cpu_count()
    p = mp.Pool(processes=no_threads)
    accs = p.map(train, params)
        
    for i in range(len(params)):
        plt.figure()
        plt.plot(range(epochs), accs[i][0], label='Train Accuracy')
        plt.plot(range(epochs), accs[i][1], label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if i == 0:
            plt.savefig('Q2_batch_size_64.png', bbox_inches='tight', dpi=100)
        if i == 1:
            plt.savefig('Q3_neuron_num_25.png', bbox_inches='tight', dpi=100)
        if i == 2:
            plt.savefig('Q4_decay_parameter_0.png', bbox_inches='tight', dpi=100)
        plt.show()

if __name__ == '__main__':
    main()


import pylab as plt
import multiprocessing as mp
from train import train, epochs, get_data


def main():
    data = get_data()
    params = []
    param1 = {}
    param1['weight_decay_parameter'] = 0
    param1['batch_size'] = 64 
    param1['hidden_layer_neuron_num'] = 25
    param1['required'] = 'cross-validation accuracy'
    param1['hidden_layer_num'] = 1
    param1['data'] = data
    param2 = param1.copy()
    param3 = param1.copy()
    param4 = param1.copy()
    param5 = param1.copy()
    
    params.append(param1)
    param2['weight_decay_parameter'] = 1e-3
    params.append(param2)
    param3['weight_decay_parameter'] = 1e-6
    params.append(param3)
    param4['weight_decay_parameter'] = 1e-9
    params.append(param4)
    param5['weight_decay_parameter'] = 1e-12
    params.append(param5)
    no_threads = mp.cpu_count()
    p = mp.Pool(processes=no_threads)
    accs = p.map(train, params)
    plt.figure()
    for i in range(len(params)):
        plt.plot(range(epochs), accs[i][0], label='Weight decay parameter:'
                 + str(params[i]['weight_decay_parameter']))
    plt.xlabel('Epochs')
    plt.ylabel('Cross Validation Accuracy')
    plt.legend()
    plt.savefig('Q4.png', bbox_inches='tight', dpi=100)
    plt.show()
    

if __name__ == '__main__':
    main()


import pylab as plt
import multiprocessing as mp
from train import train, epochs, divide_data_for_training_and_testing


def main():
    # divide the original data set into train and test data, and save to train_data.npy and test_data.npy
    divide_data_for_training_and_testing()

    params = []
    param = {}
    param['weight_decay_parameter'] = 1e-6
    param['batch_size'] = 32
    param['hidden_layer_neuron_num'] = 10
    param['required'] = 'train accuracy and test accuracy'
    param['hidden_layer_num'] = 1
    params.append(param)
    acc = train(param)

    plt.figure()
    plt.plot(range(epochs), acc[0], label='Train Accuracy')
    plt.plot(range(epochs), acc[1], label='Test Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Q1.png', bbox_inches='tight', dpi=100)
    plt.show()


if __name__ == '__main__':
    main()
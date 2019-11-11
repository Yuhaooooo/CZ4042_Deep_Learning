import pylab as plt
import os
from utilities import train

def main():
    methods = ['Momentum', 'RMSProp', 'Adam', 'Dropout']
    for method in methods:
        param = dict()
        param['required'] = 'accuracies using ' + method
        param['learning_rate'] = 0.001
        param['epochs'] = 5000
        param['batch_size'] = 128
        param['conv_1_feature_map_num'] = 40
        param['conv_2_feature_map_num'] = 70

        test_accuracies, training_costs = train(param)

        plt.figure()
        plt.plot(range(param['epochs']), training_costs)
        plt.xlabel('Epochs')
        plt.ylabel('Training Cost')
        path = os.path.join('.', 'others', 'figures', 'Q3_' + method + 'TrainingCost.png')
        plt.savefig(path, bbox_inches='tight', dpi=100)

        plt.figure()
        plt.plot(range(param['epochs']), test_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        path = os.path.join('.', 'others', 'figures', 'Q3_' + method + 'TestAccuracy.png')
        plt.savefig(path, bbox_inches='tight', dpi=100)

    param = dict()
    param['required'] = 'accuracies'
    param['learning_rate'] = 0.001
    param['epochs'] = 5000
    param['batch_size'] = 128
    param['conv_1_feature_map_num'] = 40
    param['conv_2_feature_map_num'] = 70

    test_accuracies, training_costs = train(param)

    plt.figure()
    plt.plot(range(param['epochs']), training_costs)
    plt.xlabel('Epochs')
    plt.ylabel('Training Cost')
    path = os.path.join('.', 'others', 'figures', 'Q3_TrainingCost.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)

    plt.figure()
    plt.plot(range(param['epochs']), test_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    path = os.path.join('.', 'others', 'figures', 'Q3_TestAccuracy.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)


if __name__ == '__main__':
    main()
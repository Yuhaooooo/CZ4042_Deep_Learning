import pylab as plt
import math
import os
import numpy as np
from utilities import train


def main():
    conv_1_feature_map_num_choices = [40, 50, 60, 70]
    conv_2_feature_map_num_choices = [40, 50, 60, 70]
    path = os.path.join('.', 'others', 'npy', 'grid_search_result_data.npy')

    # train the models to get grid search results, uncomment the below code block to retrain the models
    # grid_search_results = []
    # for i in conv_1_feature_map_num_choices:
    #     for j in conv_2_feature_map_num_choices:
    #         param = dict()
    #         param['learning_rate'] = 0.001
    #         param['epochs'] = 5000
    #         param['batch_size'] = 128
    #         print(str(i), str(j))
    #         param['conv_1_feature_map_num'] = i
    #         param['conv_2_feature_map_num'] = j
    #         param['required'] = 'accuracies'
    #         test_accuracies, training_costs = train(param)
    #         grid_search_results.append([test_accuracies, training_costs])
    # np.save(path, grid_search_results)

    # plot the accuracies
    results = np.load(path)
    index = 0
    for i in conv_1_feature_map_num_choices:
        for j in conv_2_feature_map_num_choices:
            test_accuracies = results[index][0]
            training_costs = results[index][1]
            # uncomment the below code block to plot training costs
            # plt.figure()
            # plt.plot(range(5000), training_costs)
            # plt.xlabel('Epochs')
            # plt.ylabel('Training Cost')
            # path = os.path.join('.', 'others', 'figures', 'Q2_' + str(i) + str(j) + 'TrainingCost.png')
            # plt.savefig(path, bbox_inches='tight', dpi=100)

            plt.figure()
            plt.plot(range(5000), test_accuracies)
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy')
            path = os.path.join('.', 'others', 'figures', 'Q2_' + str(i) + str(j) + 'TestAccuracy.png')
            plt.savefig(path, bbox_inches='tight', dpi=100)

            index += 1
    # found that 40/70, 60/50, 60/70 are the top three options


if __name__ == '__main__':
    main()
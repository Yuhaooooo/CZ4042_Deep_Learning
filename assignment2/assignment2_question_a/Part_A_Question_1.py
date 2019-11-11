import pylab as plt
import os
from utilities import train, NUM_CHANNELS, IMG_SIZE


def main():
    param = dict()
    param['required'] = 'accuracies with feature maps'
    param['learning_rate'] = 0.001
    param['epochs'] = 5000
    param['batch_size'] = 128
    param['conv_1_feature_map_num'] = 50
    param['conv_2_feature_map_num'] = 60

    test_accuracies, training_costs, images_to_plot = train(param)

    plt.figure()
    plt.plot(range(param['epochs']), training_costs)
    plt.xlabel('Epochs')
    plt.ylabel('Training Cost')
    path = os.path.join('.', 'others', 'figures', 'Q1_TrainingCost.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)

    plt.figure()
    plt.plot(range(param['epochs']), test_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    path = os.path.join('.', 'others', 'figures', 'Q1_TestAccuracy.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)

    plt.figure()
    plt.gray()
    image_1 = images_to_plot[0].reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    plt.imshow(image_1)
    path = os.path.join('.', 'others', 'figures', 'Q1_TestImage1.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)

    plot_feature_map(images_to_plot[2], 1, 1, 'Conv')
    plot_feature_map(images_to_plot[3], 1, 2, 'Conv')
    plot_feature_map(images_to_plot[6], 1, 1, 'Pool')
    plot_feature_map(images_to_plot[7], 1, 2, 'Pool')

    plt.figure()
    plt.gray()
    image_2 = images_to_plot[1].reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    plt.imshow(image_2)
    path = os.path.join('.', 'others', 'figures', 'Q1_TestImage2.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)

    plot_feature_map(images_to_plot[4], 2, 1, 'Conv')
    plot_feature_map(images_to_plot[5], 2, 2, 'Conv')
    plot_feature_map(images_to_plot[8], 2, 1, 'Pool')
    plot_feature_map(images_to_plot[9], 2, 2, 'Pool')


def plot_feature_map(maps, image_num, layer_num, layer_name):
    filters = maps.shape[3]
    if layer_num == 1:
        plt.figure(figsize=(30, 16))
    else:
        plt.figure(figsize=(30, 20))
    columns_num = 10
    rows_num = filters // 10
    plt.tight_layout()
    for i in range(filters):
        plt.subplot(rows_num, columns_num, i+1)
        plt.title('Filter ' + str(i + 1))
        plt.imshow(maps[0, :, :, i], interpolation="nearest", cmap="gray")
    path = os.path.join('.', 'others', 'figures', 'Q1_Image' + str(image_num) + layer_name + str(layer_num) + '.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)


if __name__ == '__main__':
    main()

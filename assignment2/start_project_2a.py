import numpy as np

e = np.save('data', [1,2,3,4,5,6,7,8])

filters = maps.shape[3]
plt.figure()
columns_num = 5
if layer_num == 1:
    rows_num = 10
elif layer_num == 2:
    rows_num = 12
else:
    rows_num = 0
plt.tight_layout(rect=[0, 0, 0, 1])
for i in range(filters // 2):
    plt.subplot(rows_num // 2, columns_num, i + 1)
    plt.title('Filter ' + str(i + 1))
    plt.imshow(maps[0, :, :, i], interpolation="nearest", cmap="gray")
    path = os.path.join('.', 'others', 'figures', 'Q1_Image' + str(image_num) + 'Conv' + str(layer_num) + 'I.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)
for i in range(filters // 2, filters):
    plt.subplot(rows_num // 2, columns_num, i - filters // 2 + 1)
    plt.title('Filter ' + str(i + 1))
    plt.imshow(maps[0, :, :, i], interpolation="nearest", cmap="gray")
    path = os.path.join('.', 'others', 'figures', 'Q1_Image' + str(image_num) + 'Conv' + str(layer_num) + 'II.png')
    plt.savefig(path, bbox_inches='tight', dpi=100)
from others.utensils import *

X_train, X_test, y_train, y_test = load_data(0) # 0 means original data (without removing features)

lr = 1e-3
decay = 1e-3
batch_size = 8
epochs = 10000
epochs_interval =100
early_stop_threshold = 1e-2

model = Sequential([
    Dense(10, input_dim=X_train.shape[1], activation='relu', kernel_initializer='lecun_normal', kernel_regularizer=l2(decay)),
    Dense(y_train.shape[1], kernel_initializer='he_normal', kernel_regularizer=l2(decay))
    ])

model.compile(optimizer=optimizers.SGD(lr=lr),
                  loss='mse')

train_loss=[100]
val_loss=[100]
early_stop=0

for i in range(epochs//epochs_interval):
    h = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_interval, verbose=0, validation_data=(X_test, y_test), shuffle=True)
    train_loss.extend(h.history['loss'])
    val_loss.extend(h.history['val_loss'])
    print(epochs_interval*(i+1),val_loss[-1])
    
#     if val loss increase less than early stop thrshold, then stop
    if val_loss[-1] * (1+early_stop_threshold) > val_loss[-100] and early_stop==0:
        early_stop = epochs_interval*(i+1)
        print('\n\nnow converge... store the early stopping epoch...\n\n')
        

plt.figure()
plt.plot(train_loss[1:])
plt.plot(val_loss[1:])
plt.title('mse')
plt.xlabel('epoch')
plt.legend(['train_mean_squared_error', 'test_mean_squared_error',], loc='upper left')
plt.savefig('./others/plot/q1/10000epoch.png')

plt.figure()
plt.plot(train_loss[1:early_stop])
plt.plot(val_loss[1:early_stop])
plt.title('mse')
plt.xlabel('epoch')
plt.legend(['train_mean_squared_error', 'test_mean_squared_error',], loc='upper left')
plt.savefig('./others/plot/q1/earlystop.png')


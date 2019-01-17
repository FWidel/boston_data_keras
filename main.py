import matplotlib.pyplot as plt
from keras import losses
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler

def main():
     (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)
    neural_network_mnist1 = Sequential()
    neural_network_mnist1.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))
    neural_network_mnist1.add(Dropout(0.1))
    neural_network_mnist1.add(Dense(5, activation='linear'))
    neural_network_mnist1.add(Dense(1, activation='linear'))
    neural_network_mnist1.summary()
    sgd = SGD(lr=0.006)
    neural_network_mnist1.compile(optimizer=sgd, loss=losses.mean_squared_error )
    run_hist = neural_network_mnist1.fit(x_train, y_train, epochs=500,  validation_data=(x_test, y_test), verbose=True, shuffle=False)
    print("Model  train_data [loss]: ", neural_network_mnist1.evaluate(x_train, y_train))
    print("Model   test_data [loss]: ", neural_network_mnist1.evaluate(x_test, y_test))
    neural_network_mnist = Sequential()
    neural_network_mnist.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))
    neural_network_mnist.add(Dense(5, activation='relu'))
    neural_network_mnist.add(Dense(1, activation='linear'))
    neural_network_mnist.summary()
    sgd = SGD(lr=0.004)
    neural_network_mnist.compile(optimizer=sgd, loss=losses.mean_squared_error)
    run_hist = neural_network_mnist.fit(x_train, y_train, epochs=500,   validation_data=(x_test, y_test), verbose=True, shuffle=False)
    print("Training neural network without dropouts..\n")
    print("Model  train_data [loss]: ", neural_network_mnist.evaluate(x_train, y_train))
    print("Model  test_data [loss]: ", neural_network_mnist.evaluate(x_test, y_test))
    plt.plot(run_hist.history["loss"], 'r', marker='.', label="Train Loss")
    plt.plot(run_hist.history["val_loss"], 'b', marker='.', label="Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == "__main__":
    main()
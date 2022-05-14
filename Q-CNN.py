import cirq, sympy, time
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

# Dataset preparation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizing the data between range of 0 and 1
x_train = x_train[..., np.newaxis] / 255.0
x_test = x_test[..., np.newaxis] / 255.0

print('Number of training images: ', len(x_train))
print('Number of test images: ', len(x_test))


# This function allows for the classification of all 10 classes (0-9) as 0 or 1 respectively.
# If the output is true during inferencing, it will be 1. Otherwise, 0.
def data_filter(x, y):
    # Applied a filter to training and testing sets.
    # np.where creates a list of indices where 0 and 1 labels appear.
    # If we add these as our x and y lists, We get the data that hold the labels 0 and 1
    x = x[np.where((y == 1) | (y == 0))]
    y = y[np.where((y == 1) | (y == 0))]

    y = y == 1

    return x, y


# These lists only have the labels 3 and 6
x_train, y_train = data_filter(x_train, y_train)
x_test, y_test = data_filter(x_test, y_test)

print('Number of filtered training images: ', len(x_train))
print('Number of filtered test images: ', len(x_test))

# The images in the MNIST dataset as 28x28. Due to the limitations of the
# qubit amount, we have to reduce the image size to 6x6.
x_train_small = tf.image.resize(x_train, (6, 6)).numpy()
x_test_small = tf.image.resize(x_test, (6, 6)).numpy()


# This function changes the labels to -1.0 and 1.0 to be able to use in a Parametrized Quantum Circuit layer
def convert_label(y):
    if y:
        return 1.0
    else:
        return -1.0


# Converting the train and test files for the y-axis to fall within the range of -1.0 to 1.0
y_train_converted = [convert_label(y) for y in y_train]
y_test_converted = [convert_label(y) for y in y_test]

# Selecting a subset for training from our minimized images
x_train_subset = x_train_small[:300]
x_test_subset = x_test_small[:100]

# Taking a subsample of 300 images
y_train_subset = y_train_converted[:300]
y_test_subset = y_test_converted[:100]


# Building the Quantum Model
def one_qubit_unitary(bit, symbols):
    # This circuit enacts a rotation of bloch sphere about the X, Y and Z axis, that depends on the values in 'symbols'
    return cirq.Circuit(cirq.X(bit) ** symbols[0], cirq.Y(bit) ** symbols[1], cirq.Z(bit) ** symbols[2])


def two_qubit_unitary(bit, symbols):
    # Creates a circuit that generates a random two qubit unitary
    circ = cirq.Circuit()
    circ += one_qubit_unitary(bit[0], symbols[0:3])
    circ += one_qubit_unitary(bit[1], symbols[3:6])

    circ += [cirq.ZZ(*bit) ** symbols[6]]
    circ += [cirq.YY(*bit) ** symbols[7]]
    circ += [cirq.XX(*bit) ** symbols[8]]

    circ += one_qubit_unitary(bit[0], symbols[9:12])
    circ += one_qubit_unitary(bit[1], symbols[12:])

    return circ


# Creating a 4x4 grid of qubits. These will be the qubits we will use.
qubits = cirq.GridQubit.rect(4, 4)

# Printing out the qubits line by line
for i, j in zip(qubits[0::2], qubits[1::2]):
    print(i, j)


# We represent the convolutional layer that our model will pass through as a circuit.
def convolutional_circuit(bit, symbols):
    circuit = cirq.Circuit()
    image = np.array(bit).reshape((4, 4))

    for j in range(0, 4, 2):
        for k in range(0, 4, 2):
            circuit += two_qubit_unitary([image[j, k], image[j, k + 1]], symbols)
            circuit += two_qubit_unitary([image[j + 1, k], image[j + 1, k + 1]], symbols)

    return circuit


readout = cirq.Z(qubits[-1])


@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.map_fn(lambda x: 1.0 if x >= 0 else -1.0, y_pred)
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))


# We create our model and wrap it with the PQC layer.
def create_model(qubits):
    model_circuit = cirq.Circuit()
    symbols = sympy.symbols('qconv0:15')
    model_circuit += convolutional_circuit(qubits, symbols)
    return model_circuit


# ENCODING IMAGES INTO QUANTUM CIRCUITS
############################################################
# This function creates the necessary circuits to store each of our images in qubits.
def encode_circuit_16(values):
    im4 = values[1:5, 1:5]
    phi = np.ndarray.flatten(im4)
    encode_circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(4, 4)

    for i in range(16):
        encode_circuit.append(cirq.ry(np.pi * phi[i])(qubits[i]))

    return encode_circuit


x_train_16 = [encode_circuit_16(x) for x in x_train_subset]
x_test_16 = [encode_circuit_16(x) for x in x_test_subset]

x_train_tensor_16 = tfq.convert_to_tensor(x_train_16)
x_test_tensor_16 = tfq.convert_to_tensor(x_test_16)


# Custom accuracy functions to compute the output of Quantum training
@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.map_fn(lambda x: 1.0 if x >= 0 else -1.0, y_pred)
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


# COMPILATION OF THE MODEL
model = tf.keras.Sequential(
    [tf.keras.layers.Input(shape=(), dtype=tf.string),
     tfq.layers.PQC(create_model(qubits), readout)])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.Hinge(), metrics=[hinge_accuracy])

print(model.summary())  # Printing the summary

start_time = time.time()  # Setting a timer to see how long training takes.

history = model.fit(x=x_train_tensor_16,
                    y=np.asarray(y_train_subset),
                    batch_size=16,
                    epochs=25,
                    verbose=1,
                    validation_data=(x_test_tensor_16, np.asarray(y_test_subset)))

duration = '%.2f' % (time.time() - start_time)
print(f"Model training took {duration} seconds.")  # The model takes a little over 2 minutes to train

results = model.evaluate(x_test_tensor_16, np.asarray(y_test_subset))  # Evaluating the model with the test tensor

accuracy = '%.2f' % (max(results))
print(accuracy)  # 81 % accuracy
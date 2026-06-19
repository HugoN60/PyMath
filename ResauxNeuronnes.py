from Vector import Vector, dot, squared_distance
import math 
from typing import List
import random
from decsenteDeGradient import gradient_step
import tqdm

def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    """return 1 if the perceptron is triggered, else 0"""
    calculation = dot(weights, x) + bias
    return step_function(calculation)

def sigmoid(t: float) -> float:
    return 1/ (1 + math.exp(-t))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network: List[List[Vector]], input_vector: Vector) -> List[Vector]: 
    """
    Feed the neural network with the input vector,
    Return the outputs  of all the layers (not only the last).
    """
    outputs: List[Vector] = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)
        input_vector = output
    return outputs

def squerror_gradient(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """
    Consider a neural network, an input vector, and an output vector.
    Run a forward pass to obtain a prediction and backpropagate the squared error
    loss to compute gradients with respect to the neuron's weights
    """
    hidden_outputs, outputs = feed_forward(network, input_vector)

    outputs_deltas = [output * (1 - output) * (output - target)
                      for output, target in zip(outputs, target_vector)]
    
    outputs_grads = [[outputs_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                     for i, output_neuron in enumerate(network[-1])]
    
    hidden_deltas = [hidden_output * (1 - hidden_output) * 
                     dot(outputs_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    hidden_grads = [[hidden_deltas[i] * input
                    for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]
    
    return [hidden_grads, outputs_grads]


xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

network = [
    #1st layer
    [[random.random() for _ in range(2 + 1)],
     [random.random() for _ in range(2 + 1)]],
     #2nd layer
    [[random.random() for _ in range(2 + 1)]]
]

learning_rate = 1.0
for epoch in tqdm.trange(20000, desc="Neural network for XOR"):
    for x, y in zip(xs, ys):
        gradients = squerror_gradient(network, x, y)
        network = [[gradient_step(neuron, grad, -learning_rate)
                    for neuron, grad in zip(layer, layer_grad)]
                    for layer, layer_grad in zip(network, gradients)]

assert feed_forward(network, [0., 0])[-1][0] < 0.01
assert feed_forward(network, [1., 0])[-1][0] > 0.99
assert feed_forward(network, [0., 1])[-1][0] > 0.99
assert feed_forward(network, [1., 1])[-1][0] < 0.01


def fizzbuzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]
    
def binary_encode(x: int) -> Vector:
    nb = []
    for i in range(10):
        nb.append(x % 2)
        x = x // 2
    return nb
xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizzbuzz_encode(n) for n in range(101, 1024)]

NUM_HIDDEN = 25
network = [
            [[random.random() * 2 - 1 for _ in range(10+1)]
            for _ in range(NUM_HIDDEN)],
            [[random.random() * 2 - 1 for _ in range(NUM_HIDDEN + 1)]
             for _ in range(4)]]

learning_rate = 1.0
with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            predicted = feed_forward(network, x)[-1]
            epoch_loss += squared_distance(predicted, y)
            gradients = squerror_gradient(network, x, y)
            network = [[gradient_step(neuron, grad, -learning_rate)
                    for neuron, grad in zip(layer, layer_grad)]
                    for layer, layer_grad in zip(network, gradients)]
        t.set_description(f"fizzbuzz (loss: {epoch_loss:.2f})")

def argmax(xs: List[int]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])

num_correct = 0
for n in range(1, 101):
    x = binary_encode(n)
    predicted = argmax(feed_forward(network, x)[-1])
    actual = argmax(fizzbuzz_encode(n))
    labels = [str(n), "fizz", "buzz", "fizzbuzz"]
    print(n, labels[predicted], labels[actual])
    if actual == predicted:
        num_correct += 1

print(f"{num_correct}/100")
    

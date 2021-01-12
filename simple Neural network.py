from typing import Callable

Vector = [float]
ActivationFunc = Callable[[float],float]

def neuron(inputs: Vector, weights: Vector, activation_func : ActivationFunc) ->float:
    return activation_func(
        sum(
            z[0]*z[1] for z in zip([1.0]+inputs, weights)
        )
    )
#the activation function :
def step(x: float) -> float:
    return 1 if x>0 else 0

weights = [-0.25, 1, 0.45]

glen = [-0.20,0.18]
tal = [0.6,-0.41]

print("glen", neuron(glen,weights,step))
print("tal", neuron(tal,weights,step))

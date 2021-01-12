from typing import Callable

Vector = [float]
ActivationFunc = Callable[[float],float]

def neuron(inputs: Vector, weights: Vector, activation_func : ActivationFunc) ->float:
    return activation_func(
        sum(
            z[0*z[1] for z in zip([1.0]+inputs, weights)]
        )
    )
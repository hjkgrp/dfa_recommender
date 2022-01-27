'''
Gated network for energy prediction.
'''
import torch
import numpy as np
from torch import nn
from torch import Tensor


def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return torch.nn.functional.batch_norm(x, None, None, bn.weight, bn.bias, True,
                                              bn.momentum, bn.eps)
    else:
        return bn(x)


class MySoftplus(nn.Module):
    """
    Shifted Softplus.
    """
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(MySoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.softplus(input, self.beta, self.threshold) - torch.tensor(np.log(2.), dtype=torch.double)

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


class MLP(nn.Module):
    """Multiple layer fully connected perceptron neural network.
    Args:
        n_in (int): number of input nodes.
        n_out (int): number of output nodes.
        n_hidden (list of int or int, optional): number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers (int, optional): number of layers.
        activation (callable, optional): activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """

    def __init__(
        self, n_in, n_out, n_hidden=50, n_layers=2, activation=None, droprate=0.2,
    ):
        super(MLP, self).__init__()
        self.n_neurons = [n_in] + \
            [n_hidden for _ in range(n_layers-1)] + [n_out]

        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(self.n_neurons[i], self.n_neurons[i + 1],
                                 bias=True), MySoftplus(), nn.Dropout(droprate)]

        layers.append(
            nn.Linear(self.n_neurons[-2], self.n_neurons[-1], bias=True))
        self.out_net = nn.Sequential(*layers)

    def forward(self, inputs):
        """Compute neural network output.
        Args:
            inputs (torch.Tensor): network input.
        Returns:
            torch.Tensor: network output.
        """
        return self.out_net(inputs)

    
class TiledMultiLayerNN(nn.Module):
    """
    Tiled multilayer networks which are applied to the input and produce n_tiled different outputs.
    These outputs are then stacked and returned. Used e.g. to construct element-dependent prediction
    networks of the Behler-Parrinello type.
    Args:
        n_in (int): number of input nodes
        n_out (int): number of output nodes
        n_tiles (int): number of networks to be tiled
        n_hidden (int): number of nodes in hidden nn (default 50)
        n_layers (int): number of layers (default: 3)
    """

    def __init__(
        self, n_in, n_out, n_tiles, n_hidden=50, n_layers=3, activation=None, droprate=0.2,
    ):
        super(TiledMultiLayerNN, self).__init__()
        self.mlps = nn.ModuleList(
            [
                MLP(
                    n_in,
                    n_out,
                    n_hidden=n_hidden,
                    n_layers=n_layers,
                    activation=activation,
                    droprate=droprate,
                )
                for _ in range(n_tiles)
            ]
        )

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): Network inputs.
        Returns:
            torch.Tensor: Tiled network outputs.
        """
        return torch.cat([net(inputs) for net in self.mlps], dim=-1)


class ElementalGate(nn.Module):
    """
    Produces a Nbatch x Natoms x Nelem mask depending on the nuclear charges passed as an argument.
    If onehot is set, mask is one-hot mask, else a random embedding is used.
    If the trainable flag is set to true, the gate values can be adapted during training.
    Args:
        elements (set of int): Set of atomic number present in the data
        onehot (bool): Use one hit encoding for elemental gate. If set to False, random embedding is used instead.
        trainable (bool): If set to true, gate can be learned during training (default False)
    """

    def __init__(self, elements, n_out, onehot=True, trainable=False):
        super(ElementalGate, self).__init__()
        self.trainable = trainable
        self.n_out = n_out

        # Get the number of elements, as well as the highest nuclear charge to use in the embedding vector
        self.nelems = len(elements)
        maxelem = int(max(elements) + 1)

        self.gate = nn.Embedding(maxelem, self.nelems)

        # if requested, initialize as one hot gate for all elements
        if onehot:
            weights = torch.zeros(maxelem, self.nelems*self.n_out)
            for idx, Z in enumerate(elements):
                weights[Z, self.n_out * idx: self.n_out*(idx+1)] = 1.0
            self.gate.weight.data = weights

        # Set trainable flag
        if not trainable:
            self.gate.weight.requires_grad = False

    def forward(self, atomic_numbers):
        """
        Args:
            atomic_numbers (torch.Tensor): Tensor containing atomic numbers of each atom.
        Returns:
            torch.Tensor: One-hot vector which is one at the position of the element and zero otherwise.
        """
        return self.gate(atomic_numbers)


class finalMLP(nn.Module):
    def __init__(
        self, elements, n_out, droprate=0.2,
    ):
        super(finalMLP, self).__init__()
        self.fc1 = nn.Linear(len(elements)*n_out, len(elements)*n_out)
        self.fc2 = nn.Linear(len(elements)*n_out, len(elements)*n_out)
        self.fc3 = nn.Linear(len(elements)*n_out, 1)
        self.dropout1 = nn.Dropout(droprate)
        self.dropout2 = nn.Dropout(droprate)
        self.activation = MySoftplus()
        self.bn_fc1 = nn.BatchNorm1d(len(elements)*n_out)
        self.bn_fc2 = nn.BatchNorm1d(len(elements)*n_out)

    def forward(self, inputs, update_batch_stats=True):
        x0 = inputs
#         x1 = call_bn(self.bn_fc1, self.activation(
#             self.fc1(x0)), update_batch_stats)
        x1 = self.activation(self.fc1(x0))
        x1 = self.dropout1(x1)
#         x2 = call_bn(self.bn_fc2, self.activation(
#             self.fc2(x1)), update_batch_stats)
        x2 = self.activation(self.fc2(x1))
        x2 = self.dropout2(x2)
        return self.fc3(x2)


class GatedNetwork(nn.Module):
    """
    Combines the TiledMultiLayerNN with the elemental gate to obtain element specific atomistic networks as in typical
    Behler--Parrinello networks [#behler1]_.
    Args:
        nin (int): number of input nodes
        nout (int): number of output nodes
        nnodes (int): number of nodes in hidden nn (default 50)
        nlayers (int): number of layers (default 3)
        elements (set of ints): Set of atomic number present in the data
        onehot (bool): Use one hit encoding for elemental gate. If set to False, random embedding is used instead.
        trainable (bool): If set to true, gate can be learned during training (default False)
        activation (callable): activation function
    References
    ----------
    .. [#behler1] Behler, Parrinello:
       Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces.
       Phys. Rev. Lett. 98, 146401. 2007.
    """

    def __init__(
        self,
        nin,
        n_out,
        elements,
        n_hidden=50,
        n_layers=3,
        trainable=False,
        onehot=True,
        activation=None,
        droprate=0.2,
        regression=True,
    ):
        super(GatedNetwork, self).__init__()
        self.nelem = len(elements)
        self.gate = ElementalGate(
            elements, n_out=n_out, trainable=trainable, onehot=onehot)
        self.fmpl = finalMLP(elements, n_out, droprate)
        self.regression = regression
        self.network = TiledMultiLayerNN(
            nin,
            n_out,
            self.nelem,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            droprate=droprate,
        )

    def forward(self, inputs, update_batch_stats=True):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.
        Returns:
            torch.Tensor: Output of the gated network.
        """
        atomic_numbers = torch.Tensor.int(
            inputs[:, :, -1]).type(torch.LongTensor)
        representation = inputs[:, :, :-1]
        gated_network = self.gate(atomic_numbers) * \
            self.network(representation)
        # ---direct summation without feed forward---
        # return torch.sum(gated_network, dim=[-2, -1], keepdim=False)
        # ---element aggregation---
        aggre = torch.sum(gated_network, dim=[-2], keepdim=False)
        out = torch.squeeze(
            self.fmpl(aggre, update_batch_stats=update_batch_stats))
        # out = self.fmpl(aggre, update_batch_stats=update_batch_stats)
        return out

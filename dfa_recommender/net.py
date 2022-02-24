'''
Gated network for energy prediction.
'''
import torch
import numpy as np
from torch import nn
from torch import Tensor


def call_bn(bn: nn.BatchNorm1d, x: Tensor, update_batch_stats: bool = True) -> None:
    '''
    Call for batch normalization
    '''
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return torch.nn.functional.batch_norm(x, None, None, bn.weight, bn.bias, True,
                                              bn.momentum, bn.eps)
    else:
        return bn(x)


class MySoftplus(nn.Module):
    """
    Shifted Softplus such as MySoftplus(0) = 0
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
    '''
    Multiple layer fully connected neural network.
    Each type of element has a MLP.
    Same elements share the same MLP (i.e., weight sharing)
    '''

    def __init__(self, n_in: int, n_out: int,
                 n_hidden: int = 50, n_layers: int = 3,
                 droprate: float = 0.2) -> None:
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

    def forward(self, inputs: Tensor) -> Tensor:
        '''
        Compute output.

        Parameters
        ----------
        inputs: torch.Tensor,
            model input.

        Returns:
        --------
        outputs: torch.Tensor,
            model output.
        '''
        outputs = self.out_net(inputs)
        return outputs


class TiledMultiLayerNN(nn.Module):
    """
    Tiled multilayer networks.
    A list of MLPs
    These MLPs are applied to the input to which the outputs as concatenated.
    The purpose is to create element-wise prediction.
    Note that n_tiles should be the same as the number of element types in your data set.
    """

    def __init__(self, n_in: int, n_out: int, n_tiles: int,
                 n_hidden: int = 50, n_layers: int = 3,
                 droprate: float = 0.2) -> None:
        super(TiledMultiLayerNN, self).__init__()
        self.mlps = nn.ModuleList(
            [
                MLP(
                    n_in,
                    n_out,
                    n_hidden=n_hidden,
                    n_layers=n_layers,
                    droprate=droprate,
                )
                for _ in range(n_tiles)
            ]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        '''
        Compute output.

        Parameters
        ----------
        inputs: torch.Tensor,
            model input.

        Returns:
        --------
        outputs: list,
            model output as list of torch.Tensor
        '''
        outputs = torch.cat([net(inputs) for net in self.mlps], dim=-1)
        return outputs


class ElementalGate(nn.Module):
    """
    Element based masking.
    Produces a Nbatch x Natoms x Nelem mask depending on the nuclear charges passed as an argument.
    The purpose is to create element-wise activate based on the block-wise weights in self.gate
    If onehot is set, mask is one-hot mask, else a random embedding is used.
    If the trainable flag is set to true, the gate values can be adapted during training.
    It is recommended to create a mapping dictionary for your elements. For example:
    mapping = {"X": 0, "H": 1, "C": 2, "N": 3, "O": 4, "F": 5}
    Args:
        elements (set of int): Set of atomic number present in the data
        onehot (bool): Use one hit encoding for elemental gate. If set to False, random embedding is used instead.
        trainable (bool): If set to true, gate can be learned during training (default False)
    """

    def __init__(self, elements, n_out, onehot=True, trainable=False):
        super(ElementalGate, self).__init__()
        self.trainable = trainable
        self.n_out = n_out

        self.nelems = len(elements)
        maxelem = int(max(elements) + 1)

        self.gate = nn.Embedding(maxelem, self.nelems)

        if onehot:
            weights = torch.zeros(maxelem, self.nelems*self.n_out)
            for idx, Z in enumerate(elements):
                weights[Z, self.n_out * idx: self.n_out*(idx+1)] = 1.0
            self.gate.weight.data = weights

        if not trainable:
            self.gate.weight.requires_grad = False

    def forward(self, inputs: Tensor) -> Tensor:
        '''
        Compute output.

        Parameters
        ----------
        inputs: torch.Tensor,
            model input as atomic numbers

        Returns:
        --------
        outputs: torch.Tensor,
            model output which is unity at the position of the element and zero otherwise.
        '''
        outputs = self.gate(inputs)
        return outputs


class finalMLP(nn.Module):
    '''
    The final fully connected neural network that maps the outputs from ElementalGate to the final outputs.

    '''

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

    def forward(self, inputs: Tensor,
                update_batch_stats: bool = True) -> Tensor:
        '''
        Compute output.

        Parameters
        ----------
        inputs: torch.Tensor,
            model inputs
        update_batch_stats: bool, Optional, default as True
            used only in batch normalization

        Returns:
        --------
        outputs: torch.Tensor,
            model outputs.
        '''
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
    '''
    Behler-Parrinello type gated networks.
    '''

    def __init__(
        self,
        nin: int,
        n_out: int,
        elements: list,
        n_hidden: int = 50,
        n_layers: int = 3,
        trainable: bool = False,
        onehot: bool = True,
        droprate: float = 0.2,
    ):
        super(GatedNetwork, self).__init__()
        self.nelem = len(elements)
        self.gate = ElementalGate(
            elements, n_out=n_out, 
            trainable=trainable, onehot=onehot
            )
        self.fmpl = finalMLP(elements, n_out, droprate)
        self.network = TiledMultiLayerNN(
            nin,
            n_out,
            self.nelem,
            n_hidden=n_hidden,
            n_layers=n_layers,
            droprate=droprate,
        )

    def forward(self, inputs: Tensor,
                update_batch_stats: bool = True) -> Tensor:
        '''
        Compute output.

        Parameters
        ----------
        inputs: torch.Tensor,
            model inputs, [batch_size, max(natoms), :-1] are the molecule features,
            [batch_size, max(natoms), -1] encode the element type.
        update_batch_stats: bool, Optional, default as True
            used only in batch normalization

        Returns:
        --------
        outputs: torch.Tensor,
            model outputs.
        '''
        atomic_numbers = torch.Tensor.int(
            inputs[:, :, -1]).type(torch.LongTensor)
        representation = inputs[:, :, :-1]
        gated_network = self.gate(atomic_numbers) * \
            self.network(representation)
        ## ---direct summation without feed forward, original BP---
        # return torch.sum(gated_network, dim=[-2, -1], keepdim=False)
        ## ---element aggregation---
        aggre = torch.sum(gated_network, dim=[-2], keepdim=False)
        outputs = torch.squeeze(
            self.fmpl(aggre, update_batch_stats=update_batch_stats))
        return outputs

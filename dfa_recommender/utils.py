from functools import partial
import numpy as np
import torch
import sklearn.decomposition
from sklearn.manifold import TSNE
import umap


def get_latent_space(model, input):
    activations = {}
    def save_activation(name, model, input, output):
        activations.update({name: output.cpu()})

    model.fc1.register_forward_hook(partial(save_activation, "fc1"))
    model.fc2.register_forward_hook(partial(save_activation, "fc2"))
    model.fc3.register_forward_hook(partial(save_activation, "fc3"))
    with torch.no_grad():
        _ = model(input, update_batch_stats=True)
    activations = {_name: _outputs.numpy() for _name, _outputs in activations.items()}
    return activations


def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        m.bias.data.fill_(0)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        y = m.in_features
        m.weight.data.uniform_(-1.0 / np.sqrt(y), 1.0 / np.sqrt(y))
        m.bias.data.fill_(0)


def print_model(model, optimizer):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])


def do_pca(input):
    pca = sklearn.decomposition.PCA()
    pca.fit(input)
    output = pca.transform(input)
    return output


def do_tsne(input):
    tsne = TSNE(random_state=np.random.RandomState(0))
    output = tsne.fit_transform(input)
    return output


def do_umap(input):
    mapping = umap.UMAP(n_neighbors=40,
                        min_dist=0.25,
                        metric='manhattan',
                        random_state=np.random.RandomState(0)).fit(input)
    output = mapping.transform(input)
    return output


def name_assemble(kind, n_label, n_sample, hyperparams):
    name = kind + "_nlabel_%d_nsample_%d" % (n_label, n_sample)
    for key in hyperparams:
        name += "_%s_%s" % (key, str(hyperparams[key]))
    name += ".out"
    return name


def folder_assemble(kind, n_label, n_sample):
    name = kind + "/" + kind + "_nlabel_%d_nsample_%d" % (n_label, n_sample)
    return name


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

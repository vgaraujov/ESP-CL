from torch.optim import SGD
from torch.nn import Linear
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleCNN

import torchvision.models as models

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.training.plugins import ReplayPlugin, GDumbPlugin
from avalanche.training.strategies import BaseStrategy

from ESPPlugin import ESPPlugin

def get_model(model):
    if model == 'simplCNN':
        model = SimpleCNN(num_classes=10)
        name_layers = ['features.0','features.2','features.6','features.12']
    
    elif model == 'resnet':
        model = models.resnet18(pretrained=True)
        model.fc = Linear(512, 10)
        name_layers = ['layer1.1.bn2','layer2.1.bn2','layer3.1.bn2','layer4.1.bn2']

    return model, name_layers

def main(name_model, use_replay, use_gdump, use_esp):
    model, name_layers = get_model(name_model)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    benchmark = SplitCIFAR10(n_experiences=5, seed=1)

    plugins = []
    if use_replay:
        plugins.append(ReplayPlugin(mem_size=200))

    if use_gdump:
        plugins.append(GDumbPlugin(mem_size=5000))

    if use_esp:
        plugins.append(ESPPlugin(name_layers, 10, layer_flatten=-1))

    cl_strategy = BaseStrategy(
        model, optimizer, criterion, train_mb_size=128, device='cuda',
        plugins=plugins
    )

    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream))

    print(results)

if __name__ == "__main__":
    name_model = 'resnet' # 'resnet' 'simpleCNN'
    use_replay = True
    use_gdump = False
    use_esp = True

    main(name_model, use_replay, use_gdump, use_esp)
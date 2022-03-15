from avalanche.training.plugins.strategy_plugin import StrategyPlugin
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

class Highways(nn.Module):
    def __init__(self, hidden_size, num_labels, use_pool = True,
                        hidden_dropout_prob=0.2) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, num_labels)

        self.use_pool = use_pool

    def forward(self, encoder_outputs):
        if self.use_pool:
            encoder_outputs = self.avgpool(encoder_outputs)
        encoder_outputs = torch.flatten(encoder_outputs, 1)
        output_encoder = self.relu(self.pooler(encoder_outputs))
        
        logits = self.fc(self.dropout(output_encoder))

        return logits

class ESPPlugin(StrategyPlugin):
    def __init__(
        self,
        name_layers_branch,
        num_classes,
        epoch_branch_layer=1,
        layer_flatten = -1,
    ):
        super().__init__()

        assert len(name_layers_branch) > 0, "Must be at least one name block"

        self.name_layers_branch = name_layers_branch
        self.num_classes = num_classes
        self.epoch_branch_layer = epoch_branch_layer
        self.layer_flatten = layer_flatten
        
        self.branch_layers = defaultdict(dict)
        self.dict_entropy = defaultdict(dict)
        self.branch_logits = defaultdict(dict)

    def get_by_name(self, name):
        def hook_fn(m, i, o):
            self.branch_logits[name] = o 
        return hook_fn

    def get_hook_all_layers(self, model):
        for name, layer in model.named_modules():
            if name in self.name_layers_branch:
                layer.register_forward_hook(self.get_by_name(name))

    def set_branch_layer(self, name, module, device):
        if type(module) == nn.Linear:
            hidden_size = module.weight.size(0)
            use_pool = False
        if type(module) == nn.Conv2d:
            hidden_size = module.weight.size(0)
            use_pool = True
        if type(module) == nn.BatchNorm2d:
            hidden_size = module.weight.size(0)
            use_pool = True

        self.branch_layers[name] = Highways(hidden_size, self.num_classes, 
                                                use_pool).to(device)

    def train_branch_layers(self, model, criterion, optimizer, dataset, device,
                                batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        for _ in range(self.epoch_branch_layer):
            for i, batch in enumerate(dataloader):
                # get only input, target and task_id from the batch
                x, y = batch[0], batch[1]
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                _ = model(x)

                losses_highway = 0
                for n in self.name_layers_branch:
                    highway_logits = self.branch_layers[n](self.branch_logits[n])
                    highway_loss = criterion(
                                    highway_logits.view(-1, self.num_classes),
                                    y.view(-1)
                                    )
                    losses_highway += highway_loss

                losses_highway /= len(self.branch_layers)
                losses_highway.backward()

                optimizer.step()

    def entropy(self, x):
        """Calculate entropy of a pre-softmax logit Tensor"""
        exp_x = torch.exp(x)
        A = torch.sum(exp_x, dim=1)  # sum of exp(x_i)
        B = torch.sum(x * exp_x, dim=1)  # sum of x_i * exp(x_i)
        return torch.log(A) - B / A

    def assign_layer_to_block(self, model):
        i = 0
        for n,_ in model.named_parameters():
            self.dict_entropy[n] = i
            if n == self.name_layers_branch[i]:
                i += 1

    def after_train_dataset_adaptation(self, strategy, **kwargs):
        for n,m in strategy.model.named_modules():
            if n in self.name_layers_branch:
                self.set_branch_layer(n, m, strategy.device)
        
        self.assign_layer_to_block(strategy.model)
        self.get_hook_all_layers(strategy.model)
    
    def before_training_exp(self, strategy, **kwargs):
        list_parameters = []
        for v in self.branch_layers.values():
            for p in v.parameters():
                list_parameters.append(p)
        optimizer = torch.optim.SGD(list_parameters, lr=0.001, momentum=0.9)
        
        self.train_branch_layers(
            strategy.model,
            strategy._criterion,
            optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
        )

    def before_update(self, strategy, **kwargs):
        x, y = strategy.mbatch[0], strategy.mbatch[1]
        x, y = x.to(strategy.device), y.to(strategy.device)
        
        entropy_s = torch.zeros(len(self.branch_layers))
        for j,n in enumerate(self.name_layers_branch):
            highway_logits = self.branch_layers[n](self.branch_logits[n])
            entropy_s[j] = self.entropy(highway_logits).mean()

        soft_entropy = torch.nn.functional.softmax(entropy_s, dim=0)
        soft_entropy[ soft_entropy > 1.0 ] = 1.0
        soft_entropy = 1-soft_entropy
        
        for n,p in strategy.model.named_modules():
            if n in self.dict_entropy:
                p.grad *= soft_entropy[self.dict_entropy[n]]

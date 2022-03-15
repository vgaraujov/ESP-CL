import torch
import torch.nn as nn

from avalanche.benchmarks.classic import SplitCIFAR10

import numpy as np
import random
import argparse

from resnet import resnet18

torch.multiprocessing.set_sharing_strategy('file_system')

def get_resnet_dataloader(mem_per_tasks, only_mem, memory_set, current_training_set):
    if only_mem:
        print('Training set Resnet has {} instances'.format(len(memory_set)))
        res_loader = torch.utils.data.DataLoader(memory_set, batch_size=32, shuffle=True, num_workers=2)
    else:
        if mem_per_tasks == 0:
            res_loader = torch.utils.data.DataLoader(current_training_set, batch_size=32, shuffle=True, num_workers=2)
            print('Training set Resnet has {} instances'.format(len(current_training_set)))
        else:
            memory_add_train = torch.utils.data.ConcatDataset((memory_set, current_training_set))
            print('Training set Resnet has {} instances'.format(len(memory_add_train)))
            res_loader = torch.utils.data.DataLoader(memory_add_train, batch_size=32, shuffle=True, num_workers=2)
    
    return res_loader

def get_model(pretrained, use_higway, num_classes):
    model = resnet18(pretrained=pretrained, use_highway=use_higway, new_num_classes=num_classes)
    model.fc = nn.Linear(512 * model.block.expansion, num_classes)
    return model

def update_memory(mem_per_tasks, current_training_set, memory):
    new_memory = None
    if mem_per_tasks == 0:
        len_dataset = len(current_training_set)
        indices = random.sample(list(range(len_dataset)), int(len_dataset*0.01))
        new_memory = torch.utils.data.Subset(current_training_set, indices)

    else:
        len_dataset = len(current_training_set)
        indices = random.sample(list(range(len_dataset)), int(mem_per_tasks))
        sub_train_dataset = torch.utils.data.Subset(current_training_set, indices)

        if memory is None:
            new_memory = sub_train_dataset
        else:
            new_memory = torch.utils.data.ConcatDataset((memory, sub_train_dataset))

    return new_memory

def multiply_entropy(entropies, model):
    entropy_s = torch.zeros(len(entropies))
    for i, highway_exit in enumerate(entropies):
        entropy = highway_exit[1]
        entropy_s[i] = entropy.mean()

    soft_entropy = torch.nn.functional.softmax(entropy_s, dim=0)
    soft_entropy[ soft_entropy > 1.0 ] = 1.0
    soft_entropy = 1-soft_entropy

    dict_entropy = { "layer{}.".format(i+1): s for i, s in enumerate(soft_entropy) }
    for name,p in model.named_parameters():
        for k in dict_entropy.keys():
            if k in name:
                p.grad *= dict_entropy[k]
    
    return soft_entropy.cpu()

def train_highways(model, train_loader, num_classes, highway_penalizer, device):
    model.use_highway = True
    loss_fn = torch.nn.CrossEntropyLoss()

    for name, p in model.named_parameters():
        if 'highway' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    correct = 0.0
    total = 0.0
    loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        inputs = batch[0].to(device)#.half()
        labels = batch[1].to(device)

        output, entropies = model(inputs)

        losses_highway = 0
        for highway_outputs in entropies:
            highway_logits = highway_outputs[0]
            highway_loss = loss_fn(highway_logits.view(-1, num_classes), labels.view(-1))
            losses_highway += highway_loss
        losses_highway /= len(entropies)

        l = losses_highway*highway_penalizer
        l.backward()

        optimizer.step()
        
        _, preds = output.max(1)
        correct += preds.eq(labels.clone().view_as(preds)).sum().item()
        total += inputs.size(0)
        loss += l.item()
    
    return correct/total, loss/len(train_loader)
        
def train_resnet(model, train_loader, use_highways, freeze, device):
    loss_fn = torch.nn.CrossEntropyLoss()

    if freeze:
        for name, p in model.named_parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        for name, p in model.named_parameters():
            if 'highway' in name:
                p.requires_grad = False
            else:
                p.requires_grad = True

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    correct = 0.0
    total = 0.0
    loss = 0.0
    layer_mod = []

    for batch in train_loader:
        optimizer.zero_grad()

        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        output, entropies = model(inputs)
        l = loss_fn(output, labels)

        l.backward()

        if use_highways:
            layer_mod.append(multiply_entropy(entropies, model).cpu())
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
        _, preds = output.max(1)
        correct += preds.eq(labels.clone().view_as(preds)).sum().item()
        total += inputs.size(0)
        loss += l.item()

    return correct/total, loss/len(train_loader), layer_mod

def test(model, val_loader, device):
    loss_fn = torch.nn.CrossEntropyLoss()

    loss = 0.0
    acc = 0.0
    total = 0.0
    for batch in val_loader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        outputs, _ = model(inputs)
        loss += loss_fn(outputs, labels)

        _, preds = torch.max(outputs, 1)
        acc += (preds == labels.data).sum()
        total += len(labels)

    return acc/total, loss/len(val_loader)

def parse_train_args():
    parser = argparse.ArgumentParser("Plasticity in Resnet")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--mem_per_tasks", type=int, default=100)
    parser.add_argument("--n_tasks", type=int, default=5)
    parser.add_argument('--highway_penalizer', type=float, default=0.5)

    parser.add_argument("--use_highway", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--only_mem", action="store_true")
    parser.add_argument("--train_highway_only", action="store_true")

    args = parser.parse_args()

    return args

def scenario():
    args = parse_train_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args)

    memory_set = None
    scenario = SplitCIFAR10(
        n_experiences=args.n_tasks,
        seed=args.seed,
        )
    
    model = get_model(args.pretrained, args.use_highway, args.num_classes).to(device)

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    acc_val = torch.zeros((args.n_tasks,args.n_tasks)).cpu()
    all_mod = []

    for i, experience in enumerate(train_stream):
        print("Start of task ", experience.current_experience)
        print('Classes in this task:', experience.classes_in_this_experience)
        print("Total Elements Experience: ", len(experience.dataset))

        current_training_set = experience.dataset
        memory_set = update_memory(args.mem_per_tasks, current_training_set, memory_set)

        if args.use_highway and not (experience.current_experience == 0 and args.pretrained):
            if args.train_highway_only:
                print('Training set Highway has {} instances'.format(len(memory_set)))
                highway_loader = torch.utils.data.DataLoader(memory_set, batch_size=32, shuffle=True, num_workers=2)
            else:
                print('Training set Highway has {} instances'.format(len(current_training_set)))
                highway_loader = torch.utils.data.DataLoader(current_training_set, batch_size=32, shuffle=True, num_workers=2)
            
            model.train(True)
            for _ in range(args.epochs):
                tacc, tloss = train_highways(model, highway_loader, args.num_classes, args.highway_penalizer, device)
                print("Highway Train: {} acc | {} loss".format(tacc, tloss))
            model.train(False)

        if args.use_highway and experience.current_experience == 0 and not args.pretrained:
            res_loader = get_resnet_dataloader(0, args.only_mem, memory_set, current_training_set)
        else:
            res_loader = get_resnet_dataloader(args.mem_per_tasks, args.only_mem, memory_set, current_training_set)

        for _ in range(args.epochs):
            model.train(True)
            tacc, tloss, layer_mod = train_resnet(model, res_loader, args.use_highway, args.freeze, device)
            all_mod.append(layer_mod)
            model.train(False)
            print('LOSS train {} | ACC train {}'.format(tloss, tacc))

    vacc, vloss = 0, 0
    for j in range(i+1): # args.n_tasks
        val_loader = torch.utils.data.DataLoader(test_stream[j].dataset, batch_size=16, shuffle=False, num_workers=2)
        outs = test(model, val_loader, device)
        vacc += outs[0]
        vloss += outs[1]
        acc_val[i,j] = outs[0]

    print('LOSS valid {} | ACC valid {}\n{}'.format(vloss/(j+1), vacc/(j+1), acc_val))

    
    name_file = "results/{}_{}_{}_{}_{}_{}_{}_{}.pth".format(args.use_highway, args.freeze, args.pretrained,
                                    args.mem_per_tasks, args.epochs, args.seed, args.only_mem, args.ewc)
    results = {
        'acc_val': acc_val,
        'args': args,
        'layer_mod': all_mod,
    }
    torch.save(results, name_file)

if __name__ == "__main__":
    scenario()
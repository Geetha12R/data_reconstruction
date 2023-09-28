import torch
import torchvision.datasets
import torchvision.transforms

def load_bound_dataset(dataset, batch_size, shuffle=False, start=None, end=None, **kwargs):
    def _bound_dataset(dataset, start, end):
        if start is None:
            start = 0
        if end is None:
            end = len(dataset)
        return torch.utils.data.Subset(dataset, range(start, end))
    # dataset = _bound_dataset(dataset, start, end)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_cifar10(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root, train=train, transform=transform, target_transform=target_transform, download=True)
    return dataset

# root=args.datasets_dir, batch_size=128, train=True, shuffle=False, start=0, end=50000)



def move_to_type_device(x, y, device):
    print('X:', x.shape)
    print('y:', y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    x, y = x.to(device), y.to(device)
    return x, y

def create_labels(y0):
    labels_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    y0 = torch.stack([torch.tensor(labels_dict[int(cur_y)]) for cur_y in y0])
    return y0

def get_balanced_data(args, data_loader, data_amount, batch_size):
    print('BALANCING DATASET...')
        # get balanced data
    data_amount_per_class = data_amount // args.num_classes

    labels_counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    x0, y0 = [], []
    got_enough = False
    for bx, by in data_loader:
        # by = create_labels(by)
        for i in range(len(bx)):
            if labels_counter[int(by[i])] < data_amount_per_class:
                labels_counter[int(by[i])] += 1
                x0.append(bx[i])
                y0.append(by[i])
            if all(count >= data_amount_per_class for count in labels_counter.values()):
                got_enough = True
                break
        if got_enough:
            break
    x0, y0 = torch.stack(x0), torch.stack(y0)
    for i in range(0,10):
        print(f'{i}: {y0[y0 == i].shape[0]}')
    x0 = x0.to(torch.get_default_dtype())
    y0 = y0.to(torch.get_default_dtype())
    x0, y0 = x0.to(args.device), y0.to(args.device)
    # x0, y0 = move_to_type_device(x0, y0, 'cpu')
    # dataset = torch.utils.data.TensorDataset(x0, y0)
    # dataloader =  torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # return data_loader
    return [(x0,y0)]


def load_cifar10(root, batch_size, train=False, transform=None, target_transform=None, **kwargs):
    transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # dataset = fetch_cifar10(root, train, transform, target_transform)
    dataset = torchvision.datasets.CIFAR10(root, train=train, transform=transform, download=True)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=0)
    return data_loader

def load_cifar10_data(args):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
    # Get Train Set
    print('TRAINSET BALANCED')
    train_loader = load_cifar10(root=args.datasets_dir, batch_size=128, train=True, shuffle=False, start=0, end=50000)
    train_loader = get_balanced_data(args, train_loader, args.data_amount, batch_size=128)

    # Get Test Set (balanced)
    print('LOADING TESTSET')
    assert not args.data_use_test or (args.data_use_test and args.data_test_amount >= 2), f"args.data_use_test={args.data_use_test} but args.data_test_amount={args.data_test_amount}"
    test_loader = load_cifar10(root=args.datasets_dir, batch_size=128*2, train=False, shuffle=False, start=0, end=10000)
    test_loader = get_balanced_data(args, test_loader, args.data_test_amount, batch_size=128*2)

    # move to cuda and double
    # x0, y0 = move_to_type_device(x0, y0, args.device)
    # x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)
    # print(f'BALANCE:')
    # for i in range(0,10):
    #     print(f'{i}: {y0[y0 == i].shape[0]}')

    # return [(x0, y0)], [(x0_test, y0_test)], None
    return train_loader, test_loader, None

def get_dataloader(args):
    args.input_dim = 32 * 32 * 3
    args.num_classes = 10
    args.output_dim = 10
    args.dataset = 'cifar10'

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes 
    args.data_use_test = True
    args.data_test_amount = args.data_per_class_test * args.num_classes
    print('Train Data amount ',args.data_amount)
    print('Test Data amount ',args.data_test_amount)


    data_loader = load_cifar10_data(args)
    return data_loader


    # labels_counter = {}

    # x0, y0 = [], []
    # got_enough = False

    # for bx, by in data_loader:
    #     for i in range(len(bx)):
    #         flag = by[i] not in labels_counter
    #         if flag or labels_counter[by[i]] < data_amount_per_class:
    #             if flag:
    #                 labels_counter[by[i]] = 1
    #             else:
    #                 labels_counter[(by[i])] += 1
    #             x0.append(bx[i])
    #             y0.append(by[i])
    #         else:
    #             got_enough = True
    #     if got_enough:
    #         break

    # x0, y0 = torch.stack(x0), torch.stack(y0)
    # return x0, y0
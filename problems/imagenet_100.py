import torch
import torchvision.datasets
import torchvision.transforms as transforms


def load_bound_dataset(dataset, batch_size, shuffle=False, start=None, end=None, **kwargs):
    def _bound_dataset(dataset, start, end):
        if start is None:
            start = 0
        if end is None:
            end = len(dataset)
        return torch.utils.data.Subset(dataset, range(start, end))

    dataset = _bound_dataset(dataset, start, end)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_imagenet100(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else transforms.Compose([
        transforms.Resize((224, 224)),
            # Random horizontal flip
        transforms.RandomHorizontalFlip(0.5),
            # Random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = torchvision.datasets.ImageFolder(
        root=root,
        transform=transform
    )
    return dataset


def load_imagenet100(root, batch_size, train=False, transform=None, target_transform=None, **kwargs):
    dataset = fetch_imagenet100(root, train, transform, target_transform)
    return load_bound_dataset(dataset, batch_size, **kwargs)


def move_to_type_device(x, y, device):
    print('X:', x.shape)
    print('y:', y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    x, y = x.to(device), y.to(device)
    return x, y


def get_balanced_data(args, data_loader, data_amount):
    print('BALANCING DATASET...')
    # get balanced data
    data_amount_per_class = data_amount // args.num_classes

    labels_counter = {}

    x0, y0 = [], []
    for bx, by in data_loader:
        got_enough = False
        for i in range(len(bx)):
            flag = by[i] not in labels_counter
            if flag or labels_counter[by[i]] < data_amount_per_class:
                if flag:
                    labels_counter[by[i]] = 1
                else:
                    labels_counter[(by[i])] += 1
                x0.append(bx[i])
                y0.append(by[i])
            else:
                got_enough = True
        if got_enough:
            break

    x0, y0 = torch.stack(x0), torch.stack(y0)
    return x0, y0


def load_imagenet100_data(args):
    # Get Train Set
    print('TRAINSET BALANCED')
    data_loader = load_imagenet100(root=args.imagenet_datasets_dir, batch_size=8, train=True, shuffle=True, start=0, end=500)
    x0, y0 = get_balanced_data(args, data_loader, args.data_amount)

    # Get Test Set (balanced)
    print('LOADING TESTSET')
    # assert not args.data_use_test or (args.data_use_test and args.data_test_amount >= 2), f"args.data_use_test={args.data_use_test} but args.data_test_amount={args.data_test_amount}"
    data_loader = load_imagenet100(root=args.imagenet_datasets_dir, batch_size=8, train=False, shuffle=False, start=0, end=250)
    x0_test, y0_test = get_balanced_data(args, data_loader, args.data_test_amount)

    # move to cuda and double
    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)

    print(f'BALANCE: 0: {y0[y0 == 0].shape[0]}, 1: {y0[y0 == 1].shape[0]}')

    return [(x0, y0)], [(x0_test, y0_test)], None


def get_dataloader(args):
    # to be replaced
    args.input_dim = 224 * 224 * 3
    args.num_classes = 10
    args.output_dim = 10
    args.dataset = 'imagenet100'

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    data_loader = load_imagenet100_data(args)
    return data_loader
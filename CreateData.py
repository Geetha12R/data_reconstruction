
#api to retrieve the data to be loaded
def setup_problem(args):
    if False:
        pass
    elif args.problem == 'cifar10_vehicles_animals':
        from problems.cifar10_vehicles_animals import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'mnist_odd_even':
        from problems.mnist_odd_even import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'imagenet100':
        from problems.imagenet_100 import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'cifar10_vehicles_animals_multi':
        from problems.cifar10_vehicles_animals_multi import get_dataloader
        return get_dataloader(args)
    else:
        raise ValueError(f'Unknown args.problem={args.problem}')
    return data_loader


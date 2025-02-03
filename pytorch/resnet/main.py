import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np


try:
    LOCAL_RANK: int = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE: int = int(os.environ['WORLD_SIZE'])
    WORLD_RANK: int = int(os.environ['RANK'])

except KeyError:
    raise KeyError('Please set correct environment variables')


def set_random_seeds():

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def build_model() -> nn.Module:

    device = torch.device("cuda:{}".format(LOCAL_RANK))

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
    )

    if resume == True: # noqa
        map_location = {"cuda:0": "cuda:{}".format(LOCAL_RANK)}
        ddp_model.load_state_dict(
            torch.load(model_filepath, map_location=map_location)
        )

    return ddp_model


def evaluate(model: nn.Module, device, test_loader) -> float:

    model.eval()

    correct: int = 0
    total: int = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs: torch.Tensor = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def run():

    ddp_model = build_model()

    device = torch.device("cuda:{}".format(LOCAL_RANK))

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=15,
        pin_memory=True,
    )

    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=15,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    for epoch in range(num_epochs):

        print("Local Rank: {}, Epoch: {}, Training ...".format(LOCAL_RANK, epoch))

        ddp_model.train()

        total_loss = []
        for index, (data) in enumerate(train_loader):
            print(f"Local Rank: {LOCAL_RANK}, index: {index}", end="\r")

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        print("Local Rank: {}, Epoch: {}, Loss: {}".format(LOCAL_RANK, epoch, np.mean(total_loss)))

        if epoch % 10 == 0:
            if LOCAL_RANK == 0:
                accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                torch.save(ddp_model.state_dict(), model_filepath)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        print(f'Epoch {epoch} completed')


def init_processes(backend: str):
    dist.init_process_group(backend)
    try:
        run()
    finally:
        # Ensure the process group is destroyed
        dist.destroy_process_group()


if __name__ == "__main__":

    default_backend = 'nccl'
    model_dir_default: str = "saved_models"
    model_filename_default: str = "resnet_distributed.pth"

    learning_rate_default: float = 0.1
    num_epochs_default: int = 100
    batch_size_default: int = 128
    random_seed_default: float = 0

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs",
                        type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.",
                        default=batch_size_default)
    parser.add_argument("--learning_rate", type=float,
                        help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int,
                        help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str,
                        help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str,
                        help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from saved checkpoint.")
    argv = parser.parse_args()

    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume

    model_filepath = str(os.path.join(model_dir, model_filename))

    set_random_seeds()
    init_processes(backend=default_backend)

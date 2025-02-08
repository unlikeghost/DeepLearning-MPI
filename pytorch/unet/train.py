import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import random
import numpy as np
from typing import Tuple
from tqdm import tqdm
from datetime import datetime
import time

from model import UNet
from data_loading import CarvanaDataset


def get_environment_variables() -> Tuple[int, int, int]:
    """Retrieve distributed environment variables."""
    try:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        world_rank = int(os.environ['RANK'])
        return local_rank, world_size, world_rank
    except KeyError as e:
        raise RuntimeError("Missing required environment variables for distributed training") from e


def get_system_information() -> str:
    """Retrieve system and environment setup details."""
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return f"World size: {world_size}, Local rank: {local_rank}, GPU: {torch.cuda.get_device_name(local_rank)}"


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Deterministic mode may not be optimal for performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_log_file() -> str:
    """Create a unique log file based on the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    log_file = os.path.join(
        'logs',
        f"training_log_{timestamp}.log"
    )
    return log_file


def log_to_file(filepath: str, message: str) -> None:
    """Log a message to a file."""
    with open(filepath, "a") as f:
        f.write(message + "\n")


def build_model(local_rank: int, resume: bool, model_filepath: str) -> nn.Module:
    """Build and initialize the model for training."""
    device = torch.device(f"cuda:{local_rank}")

    model = UNet(out_classes=1)
    model.to(device)

    # # Wrap the model in DistributedDataParallel
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    if resume:
        map_location = {f"cuda:{local_rank}": f"cuda:{local_rank}"}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))
    return ddp_model


def build_data_loaders(
        batch_size: int,
        local_rank: int,
) -> Tuple[DataLoader, DataLoader]:
    """Set up data loaders for training and testing."""

    dataset = CarvanaDataset(images_dir=os.path.join('data', 'images'),
                             mask_dir=os.path.join('data', 'masks'),
                             scale=0.2)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    loader_args = dict(
        batch_size=batch_size,
        num_workers=int(os.cpu_count() // 2),
        pin_memory=True,
    )

    train_sampler = DistributedSampler(train_dataset, rank=local_rank)

    train_loader = DataLoader(train_dataset, **loader_args, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, **loader_args, shuffle=False)

    return train_loader, test_loader


def evaluate_model(model: nn.Module, device: torch.device, test_loader: DataLoader) -> float:
    """Evaluate the model and return its accuracy."""
    model.eval()
    dice_scores = []  # Store dice scores for each batch

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = batch["image"].to(device, dtype=torch.float32)
            true_masks = batch["mask"].unsqueeze(1).to(device, dtype=torch.float32)

            # Get predictions
            predicted_mask = model(images)
            predicted_mask = torch.sigmoid(predicted_mask)
            predicted_mask = (predicted_mask > 0.5).float()

            # Compute Dice Score
            intersection = (predicted_mask * true_masks).sum(dim=(2, 3))
            union = predicted_mask.sum(dim=(2, 3)) + true_masks.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            dice_scores.append(dice.mean().item())

    # Overall average Dice Score
    return sum(dice_scores) / len(dice_scores)


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        num_epochs: int,
        model_filepath: str,
        local_rank: int,
        learning_rate: float = 0.1,) -> None:
    """Train the model."""

    # Unique log file for this training session
    if local_rank == 0:
        print(f"Logging training progress to: {log_file}")
        log_to_file(log_file, f"Started training at {datetime.now()}")

    # Define loss and optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()


    # Train model
    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for batch in loop:
            images = batch["image"].to(device, dtype=torch.float32)
            true_masks = batch["mask"].unsqueeze(1).to(device, dtype=torch.float32)

            predicted_masks = model(images)

            loss = criterion(predicted_masks, true_masks)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())
        print(f'Epoch {epoch + 1} finished with loss: {epoch_loss:.4f}')

        # Save time by epoch
        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_log_message = f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Duration: {epoch_duration:.2f}s"
        if local_rank == 0:
            log_to_file(log_file, epoch_log_message)

        if epoch % 10 == 0:
            if local_rank == 0:
                accuracy = evaluate_model(model, device, test_loader)
                torch.save(model.state_dict(), model_filepath)
                print("-" * 75)
                print(f'Epoch {epoch + 1} accuracy: {accuracy:.4f}')
                print("-" * 75)

def initialize_processes(
        backend: str,
        local_rank: int,
        world_size: int,
        batch_size: int,
        num_epochs: int,
        model_filepath: str,
        learning_rate: float,
        resume: bool = False):
    """Initialize distributed processes and start training."""
    dist.init_process_group(backend, rank=local_rank, world_size=world_size)

    try:

        # Build data loaders
        train_loader, test_loader = build_data_loaders(
            batch_size=batch_size,
            local_rank=local_rank,
        )

        # Build model
        model = build_model(local_rank, resume, model_filepath)
        device = torch.device(f"cuda:{local_rank}")

        print('Model built. Starting training.')
        train_model(model, train_loader, test_loader, device, num_epochs, model_filepath, local_rank, learning_rate)

    finally:
        dist.destroy_process_group()


def main(p_args: argparse.Namespace) -> None:
    model_filepath = str(os.path.join(p_args.model_dir, p_args.model_filename))
    set_random_seeds(p_args.random_seed)

    local_rank, world_size, _ = get_environment_variables()

    initialize_processes(
        backend="nccl",
        local_rank=local_rank,
        world_size=world_size,
        batch_size=p_args.batch_size,
        num_epochs=p_args.num_epochs,
        model_filepath=model_filepath,
        learning_rate=p_args.learning_rate,
        resume=p_args.resume
    )


if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise SystemError(
            "You should have a GPU available to run this script."
        )

    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        raise OSError(
            "The 'data' directory does not exist. Please create it before running the script."
        )

    if not os.path.exists(os.path.join(os.getcwd(), "logs")):
        raise OSError(
            "The 'logs' directory does not exist. Please create it before running the script."
        )

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--num_epochs", type=int,
        default=100, help="Number of training epochs."
    )

    parser.add_argument(
        "--batch_size", type=int,
        default=128, help="Batch size per process."
    )

    parser.add_argument(
        "--learning_rate", type=float,
        default=0.1, help="Learning rate."
    )

    parser.add_argument(
        "--random_seed", type=int,
        default=42, help="Seed for reproducibility."
    )

    parser.add_argument(
        "--model_dir", type=str,
        default="saved_models", help="Directory to save model."
    )

    parser.add_argument(
        "--model_filename", type=str,
        default="model.pth", help="Model filename."
    )

    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from a checkpoint."
    )

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.model_dir)):
        raise OSError(
            "The model directory does not exist. Please create it before running the script."
        )

    log_file = create_log_file()

    log_to_file(log_file, f"Batch size: {args.batch_size}")
    log_to_file(log_file, f"Number of workers: {os.cpu_count()}")
    log_to_file(log_file, f"Learning rate: {args.learning_rate}")
    log_to_file(log_file, f"Number of epochs: {args.num_epochs}")
    log_to_file(log_file, get_system_information())

    main(p_args=args)

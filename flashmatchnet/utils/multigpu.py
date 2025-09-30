import os
import torch
import torch.distributed as dist

def setup_distributed():
    """Initialize distributed training environment"""

    # Check if distributed training is available
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Initialize the process group
        dist.init_process_group(backend='nccl')

        # Set CUDA device
        torch.cuda.set_device(local_rank)

        print(f"Distributed training initialized: Rank {rank}/{world_size}, Local rank: {local_rank}")
        return True, rank, world_size, local_rank
    else:
        print("Running in single GPU mode")
        return False, 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


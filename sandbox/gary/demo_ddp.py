import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, size):
    print(f"rank {rank} size {size}")
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor, dst=1)
    else:
        dist.recv(tensor, src=0)

    print(f"rank {rank} tensor {tensor.item()}")


def init_process(rank, world_size, fn, backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size)


def main():
    world_size = 2
    processes = []

    mp.set_start_method("spawn")

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run, "gloo"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

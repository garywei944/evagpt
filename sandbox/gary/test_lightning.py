import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L


class LargeMLP(nn.Module):
    """3-layer MLP with configurable hidden size for testing precision."""

    def __init__(self, input_size: int = 1024, hidden_size: int = 16384, output_size: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


def main():
    fabric = L.Fabric(
        accelerator="cuda",
        devices=6,
        strategy="ddp",
        precision="16-mixed",
    )
    fabric.launch()
    fabric.seed_everything(42)

    # ~1B parameters: hidden_size=16384 gives roughly 1.07B params
    with fabric.init_module():
        model = LargeMLP(input_size=1024, hidden_size=16384, output_size=1024)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    fabric.print(f"Total parameters: {num_params / 1e9:.2f}B")
    fabric.print(f"Expected bf16 memory from params: {num_params * 2 / 1e9:.2f} GB")

    model, optimizer = fabric.setup(model, optimizer)
    fabric.print(f"Memory after setup: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # synthetic data
    batch_size = 32
    x = torch.randn(batch_size, 1024, device=fabric.device)
    target = torch.randn(batch_size, 1024, device=fabric.device)

    for step in range(3):
        output = model(x)
        loss = F.mse_loss(output, target)

        fabric.print(f"\nStep {step}")
        fabric.print(f"  Loss: {loss.item():.4f}")
        fabric.print(f"  Output dtype: {output.dtype}")
        fabric.print(f"  Param dtype before backward: {next(model.parameters()).dtype}")

        fabric.backward(loss)

        fabric.print(f"  Param dtype after backward: {next(model.parameters()).dtype}")

        # DeepSpeed manages gradients internally, so .grad is None
        # Check grad dtype via hook or just skip
        param = next(model.parameters())
        if param.grad is not None:
            fabric.print(f"  Grad dtype: {param.grad.dtype}")
        else:
            fabric.print("  Grad dtype: managed by DeepSpeed (not accessible via .grad)")

        optimizer.step()
        optimizer.zero_grad()

        fabric.print(f"  Param dtype after step: {next(model.parameters()).dtype}")

    fabric.print("\n" + torch.cuda.memory_summary())
    fabric.save("mlp-bf16-checkpoint.ckpt", {"model": model})


if __name__ == "__main__":
    main()

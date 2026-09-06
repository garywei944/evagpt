import logging

import lightning as L
import tiktoken
import torch
from jaxtyping import Int64
from lightning.pytorch import callbacks, loggers
from torch import Tensor, optim

from evagpt import gpt2, token_stream_dataset

logger = logging.getLogger(__file__)


class OWTDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: str):
        self.train_loader = token_stream_dataset.get_dataloader(
            "data/processed/openwebtext/train.bin", block_size=1024, batch_size=2, shuffle=True
        )
        self.val_loader = token_stream_dataset.get_dataloader(
            "data/processed/openwebtext/val.bin", block_size=1024, batch_size=8, shuffle=False
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class GPT2Model(L.LightningModule):
    def __init__(self):
        super().__init__()

        tokenizer = tiktoken.encoding_for_model("gpt2")
        self.gpt_config = gpt2.GPT2Config(
            block_size=1024, vocab_size=tokenizer.n_vocab, n_layers=2, n_heads=2, n_embd=128
        )
        self.gpt = gpt2.GPT2(config=self.gpt_config)
        self.gpt = torch.compile(self.gpt)

    def training_step(self, batch: tuple[Int64[Tensor, " T"], Int64[Tensor, " T"]], batch_idx: int):
        input_ids, labels = batch
        logits, loss = self.gpt(input_ids, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=input_ids.size(0))

        return loss

    def validation_step(self, batch: tuple[Int64[Tensor, " T"], Int64[Tensor, " T"]], batch_idx: int):
        input_ids, labels = batch
        logits, loss = self.gpt(input_ids, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=input_ids.size(0))

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.gpt.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
        return optimizer


def main():
    L.seed_everything(42)
    logging.basicConfig(level=logging.DEBUG)

    dm = OWTDataModule()
    model = GPT2Model()

    trainer = L.Trainer(
        max_steps=20000,
        logger=[
            loggers.TensorBoardLogger("lightning_logs", name="gpt2_owt"),
            loggers.WandbLogger(project="gpt2_owt", name="gpt2_owt", entity="garywei944"),
        ],
        callbacks=[callbacks.LearningRateMonitor(logging_interval="step")],
        gradient_clip_val=1.0,
        precision="bf16-mixed",
        val_check_interval=500,
    )

    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    main()

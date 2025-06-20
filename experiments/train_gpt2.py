import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    Timer,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.utilities import rank_zero_info
from torchmetrics.text import Perplexity

from transformers.optimization import get_scheduler

import torch
import torch.nn.functional as F

from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import regex as re
import os

from experiments.data.language_modeling.wikitext_datamodule import WikiTextDataModule
from src.gpt2 import GPT2Config, GPT2

torch.set_float32_matmul_precision("medium")


class GPT2Module(L.LightningModule):
    def __init__(
        self,
        dm: WikiTextDataModule = None,  # type: ignore
        model_name: str = "gpt2",
        learning_rate: float = 2.5e-4,
        weight_decay: float = 0.1,
    ):
        super().__init__()

        config = GPT2Config.from_pretrained(model_name)

        self.hparams.update(asdict(config))
        self.save_hyperparameters(ignore=["dm"])
        self.dm = dm

        self.model = GPT2(config)
        self.metrics = Perplexity()

    def setup(self, stage: str):
        if stage == "fit":
            total_devices = self.trainer.num_devices * self.trainer.num_nodes
            train_batches = len(self.dm.train_dataloader()) // total_devices
            if self.trainer.max_epochs is None:
                self.train_steps = self.trainer.max_steps
            else:
                self.train_steps = train_batches * self.trainer.max_epochs

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        _, loss = self(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits, _ = self(batch)
        loss, shift_logits, shift_labels = loss_fn(logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.metrics(shift_logits, shift_labels)
        self.log(
            "val/perplexity", self.metrics, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        logits, _ = self(batch)
        loss, shift_logits, shift_labels = loss_fn(logits, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.metrics(shift_logits, shift_labels)
        self.log(
            "test/perplexity",
            self.metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        # no_decay = ["bias", "LayerNorm.weight"]
        pattern = r"^(.*\.)?(bias|ln(\d|_f)\.weight)$"

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not re.match(pattern, n)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if re.match(pattern, n)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),  # consist with deepseek V3
            eps=1e-6,
        )
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=int(self.train_steps * 0.02),
            num_training_steps=self.train_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def loss_fn(logits, labels):
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    ce_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
    )

    return ce_loss, shift_logits, shift_labels


def parse_args():
    parser = LightningArgumentParser()
    parser.add_argument("--run_ver", type=str, default="gpt2")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-wp",
        "--wandb_project",
        type=str,
        default="gpt2",
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--strategy", type=str, default="deepspeed_stage_2")
    parser.add_argument("--precision", type=str, default="16-mixed")

    parser.add_lightning_class_args(GPT2Module, "model")
    parser.add_lightning_class_args(WikiTextDataModule, "data")

    args = parser.parse_args()
    rank_zero_info(f"{args.epochs=}")

    del args.model.dm

    return args


if __name__ == "__main__":
    timestamp = datetime.now().isoformat()
    args = parse_args()
    run_name = args.run_ver
    log_dir = Path("logs") / "gpt2" / run_name

    L.seed_everything(args.seed + int(os.getenv("LOCAL_RANK", "0")))

    dm = WikiTextDataModule(**args.data)
    model = GPT2Module(dm=dm, **args.model)

    timer = Timer()
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=[
            WandbLogger(
                project=args.wandb_project,
                name=run_name,
                entity="danaus",
            ),
            # TensorBoardLogger(save_dir=log_dir / "tensorboard", name=model_name),
            CSVLogger(
                save_dir=log_dir / "csv",
                name=run_name,
            ),
        ],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            timer,
            ModelCheckpoint(
                dirpath=f"checkpoints/{run_name}",
                filename=f"{run_name}-{{epoch:02d}}-{{val/perplexity:.2f}}",
                monitor="val/perplexity",
                mode="min",
                save_top_k=1,
            ),
            ModelSummary(max_depth=-1),
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        # enable_progress_bar=False,
        # limit_train_batches=0.01,  # 1% of the training data, for debugging
        # limit_train_batches=15,  # 5 batches of the training data, for debugging
        # limit_val_batches=15,
        # limit_test_batches=15,
        # fast_dev_run=True,
        accelerator="gpu",
        devices="auto",
        strategy=args.strategy,
        precision=args.precision,
    )

    # # Sanity check: gpt2-large perplexity on wikitext-2 test = 18.5486
    # trainer.test(model, datamodule=dm)

    # trainer.validate(model, datamodule=dm)

    trainer.fit(model, datamodule=dm)
    # print(timer.time_elapsed("train"))

    # try to load the best model
    model = GPT2Module.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    trainer.test(model, datamodule=dm)

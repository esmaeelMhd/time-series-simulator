import argparse
import torch

from timesim.data.loader import generate_sine_dataset, build_dataloaders
from timesim.models import get_model
from timesim.engine.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train a model on sine data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--pred-len", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model", type=str, default="lstm")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ckpt", type=str, default="checkpoint.pth")
    return p.parse_args()


def main():
    args = parse_args()

    series = generate_sine_dataset(length=2000)
    train_loader, val_loader = build_dataloaders(series,
                                                seq_len=args.seq_len,
                                                pred_len=args.pred_len,
                                                batch_size=args.batch_size,
                                                device=args.device)

    ModelCls = get_model(args.model)
    model = ModelCls(input_dim=series.shape[1], pred_len=args.pred_len)

    trainer = Trainer(model, device=args.device)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    trainer.save(args.ckpt)


if __name__ == "__main__":
    main() 
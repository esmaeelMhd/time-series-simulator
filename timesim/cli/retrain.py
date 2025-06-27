import argparse

from pathlib import Path

import torch

from timesim.data.loader import generate_sine_dataset, build_dataloaders
from timesim.models import get_model
from timesim.engine.retrainer import Retrainer


def parse_args():
    p = argparse.ArgumentParser(description="Retrain a model checkpoint on fresh data")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--pred-len", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--model", type=str, default="lstm")
    return p.parse_args()


def main():
    args = parse_args()

    series = generate_sine_dataset(length=1000)

    train_loader, val_loader = build_dataloaders(series,
                                                seq_len=args.seq_len,
                                                pred_len=args.pred_len,
                                                batch_size=args.batch_size,
                                                device=args.device)

    ModelCls = get_model(args.model)

    retrainer = Retrainer(model_cls=lambda: ModelCls(input_dim=series.shape[1], pred_len=args.pred_len),
                          checkpoint=args.ckpt,
                          device=args.device)

    retrainer.fine_tune(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main() 
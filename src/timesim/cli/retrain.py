import argparse
from pathlib import Path
import numpy as np
import torch

from timesim.data.loader import generate_sine_dataset, build_dataloaders
from timesim.data.schema import VariableSchema
from timesim.models import get_model
from timesim.engine.retrainer import Retrainer
from timesim.utils.config import load_config
from timesim.utils.logger import create_run_dir, init_logging
from timesim.utils.plotting import save_loss_plot, compare_simulation_plot, multi_compare_simulation_plot
from timesim.utils.misc import seed_everything, resolve_device
from joblib import load, dump


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a checkpoint on new data")
    p.add_argument("--config", type=str, default="configs/base.yml",
                   help="YAML config with defaults")

    # Overrides
    p.add_argument("--ckpt", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--pred-len", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--model", type=str)

    # SEPP specific
    p.add_argument("--retrain-method", type=str, default=None, choices=["v1", "v2", "sepp"])
    p.add_argument("--sepp-h-max", type=int)
    p.add_argument("--sepp-stride", type=int)
    return p.parse_args()


def main():
    cli_args = parse_args()

    cfg = load_config(cli_args.config, cli_args)
    seed = int(cfg.get("seed") or cfg.get("misc", {}).get("seed", 42))
    deterministic = bool(cfg.get("misc", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)
    dataset_cfg = cfg.get("dataset", {})

    # ----------------------------------------------------
    # Determine dataset/model from config
    dataset_name = Path(cfg.get("dataset", {}).get("name", "sine")).name
    ckpt_path = Path(cfg["ckpt"]).resolve()
    base_run_name = ckpt_path.parent.name              # timestamp folder name
    model_folder = ckpt_path.parents[1].name           # model folder name
    run_dir = create_run_dir(dataset=dataset_name,
                             model=model_folder,
                             suffix=f"retrain-{base_run_name}")
    logger, tb_writer = init_logging(run_dir)

    # Save merged config
    import yaml
    with open(Path(run_dir)/"config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # ----------------------------------------------------
    # Dataset preparation (real CSV or sine fallback)
    # Parameters can be provided at the top level **or** inside the dataset
    # sub-section of the YAML.  Here we check both so the behaviour mirrors
    # `timesim.cli.train`.
    # ----------------------------------------------------
    seq_len = cfg.get("seq_len") or dataset_cfg.get("seq_len", 24)
    pred_len = cfg.get("pred_len") or dataset_cfg.get("pred_len", 12)
    batch_size = cfg.get("batch_size") or dataset_cfg.get("batch_size", 32)
    device = resolve_device(cfg.get("device") or cfg.get("misc", {}).get("device", "auto"))

    input_groups = cfg.get("model_io", {}).get("input_groups", ["control"])
    output_groups = cfg.get("model_io", {}).get("output_groups", ["objective"])

    if dataset_cfg.get("name", "sine") == "sine":
        series = generate_sine_dataset(length=1500)
        train_loader, val_loader = build_dataloaders(series,
                                                    seq_len=seq_len,
                                                    pred_len=pred_len,
                                                    batch_size=batch_size,
                                                    device=device,
                                                    seed=seed,
                                                    shuffle_train=bool(cfg.get("data", {}).get("shuffle_train", True)),
                                                    drop_last=bool(cfg.get("data", {}).get("drop_last", True)))
        df_raw = None
        groups = {"control": ["sine"], "exogenous": [], "objective": ["sine"]}
        schema = VariableSchema.from_groups(groups)
    else:
        from timesim.data.loader import load_csv_dataset, build_grouped_dataloaders
        df_raw = load_csv_dataset(dataset_cfg["csv"],
                                  index_col=dataset_cfg.get("index_col", "date"),
                                  parse_dates=bool(cfg.get("data", {}).get("parse_dates", True)),
                                  slice_cfg=dataset_cfg.get("slice"),
                                  engine=str(cfg.get("data", {}).get("csv_engine", "pandas")),
                                  validation_cfg=cfg.get("data", {}).get("validation", None))
        groups = dataset_cfg["variables"]
        schema = VariableSchema.from_groups(groups)

        scaler_path = Path(cfg["ckpt"]).parent/"scaler.pkl"
        existing_scaler = load(scaler_path) if scaler_path.exists() else None

        train_loader, val_loader, scaler = build_grouped_dataloaders(df_raw,
                                                                    groups,
                                                                    input_groups,
                                                                    output_groups,
                                                                    seq_len=seq_len,
                                                                    pred_len=pred_len,
                                                                    batch_size=batch_size,
                                                                    train_split=dataset_cfg.get("train_split", cfg.get("data", {}).get("train_split", 0.7)),
                                                                    split_cfg=cfg.get("data", {}).get("splits", None),
                                                                    device=device,
                                                                    seed=seed,
                                                                    shuffle_train=bool(cfg.get("data", {}).get("shuffle_train", True)),
                                                                    drop_last=bool(cfg.get("data", {}).get("drop_last", True)),
                                                                    existing_scaler=existing_scaler,
                                                                    require_full_role_mapping=bool(
                                                                        cfg.get("data", {}).get("require_full_role_mapping", True)
                                                                    ))

        # save scaler into new run dir
        dump(scaler, run_dir/"scaler.pkl")

    # compute output_cols after groups exists
    output_cols = schema.columns_for_group_names(output_groups)

    # ----------------------------------------------------
    # Determine model name string (handle nested dict in YAML)
    if isinstance(cfg.get("model"), str):
        _model_name = cfg["model"]
    else:
        _model_name = cfg.get("model", {}).get("type", "lstm")

    ModelCls = get_model(_model_name)

    def build_model_func():
        in_dim = len(schema.columns_for_group_names(input_groups)) if df_raw is not None else series.shape[1]
        out_dim = len(output_cols) if df_raw is not None else in_dim
        kwargs = dict(input_dim=in_dim, pred_len=pred_len)
        if 'out_dim' in ModelCls.__init__.__code__.co_varnames:
            kwargs['out_dim'] = out_dim
        return ModelCls(**kwargs)

    # ----------------------------------------------------
    # Pre-retrain simulation (only if real dataset present)
    real_list, before_list, starts_used = [], [], []
    if df_raw is not None:
        sim_points = cfg.get('retrain', {}).get('sim_points', 1)
        sim_horizon = cfg.get('retrain', {}).get('sim_horizon', 10*pred_len)
        from timesim.utils.simulation import simulate_autoregressive
        model_pre = build_model_func().to(device)
        state_pre = torch.load(cfg['ckpt'], map_location=device)
        if isinstance(state_pre, dict) and "model_state_dict" in state_pre:
            state_pre = state_pre["model_state_dict"]
        model_pre.load_state_dict(state_pre)
        for _ in range(sim_points):
            real, pred, sidx = simulate_autoregressive(
                model_pre,
                df_raw,
                groups,
                input_groups,
                output_groups,
                seq_len=seq_len,
                horizon=sim_horizon,
                device=device,
                scaler=scaler,
                use_symlog=bool((cfg.get("data", {}).get("symlog", {}) or {}).get("enabled", False)),
                symlog_columns=(cfg.get("data", {}).get("symlog", {}) or {}).get("columns", None),
            )
            real_list.append(real)
            before_list.append(pred)
            starts_used.append(int(sidx))

    # ----------------------------------------------------
    # Choose retrain path
    retrain_method = cli_args.retrain_method or cfg.get('retrain', {}).get('method', 'v2')
    if retrain_method == "sepp":
        from timesim.engine.sepp_trainer import SEPPTrainer
        model = build_model_func().to(device)
        state = torch.load(cfg['ckpt'], map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        h_max = cli_args.sepp_h_max or 10*pred_len
        stride = cli_args.sepp_stride or pred_len

        sepp_ckpt = run_dir/"checkpoint_retrained.pth"

        # ----------------------------------------------------
        # Build a validation loader compatible with SEPPWindowDataset
        # ----------------------------------------------------
        from timesim.data.sepp_dataset import SEPPWindowDataset
        n_total_rows = len(df_raw)
        n_train_rows = int(
            n_total_rows * float(dataset_cfg.get("train_split", cfg.get("data", {}).get("train_split", 0.7)))
        )
        # Ensure window fits: need at least seq_len + h_max rows
        val_start = max(n_train_rows - seq_len - h_max, 0)
        val_df = df_raw.iloc[val_start:]
        if len(val_df) >= seq_len + h_max:
            sepp_val_ds = SEPPWindowDataset(val_df,
                                            groups,
                                            input_groups,
                                            output_groups,
                                            seq_len=seq_len,
                                            h_max=h_max,
                                            stride=stride)
            from torch.utils.data import DataLoader
            sepp_val_loader = DataLoader(sepp_val_ds, batch_size=batch_size, shuffle=False)
        else:
            sepp_val_loader = None  # too small, fallback to None

        sim_horizon = cfg.get('retrain', {}).get('sim_horizon', 10*pred_len)
        def on_improve_cb(m, epoch):
            if df_raw is None:
                return
            after_list_local = []
            for sidx in starts_used:
                _, pred, _ = simulate_autoregressive(
                    m, df_raw, groups, input_groups, output_groups,
                    seq_len=seq_len, horizon=sim_horizon,
                    device=device, start_idx=sidx,
                    scaler=scaler,
                    use_symlog=bool((cfg.get("data", {}).get("symlog", {}) or {}).get("enabled", False)),
                    symlog_columns=(cfg.get("data", {}).get("symlog", {}) or {}).get("columns", None),
                )
                after_list_local.append(pred)
            multi_compare_simulation_plot(
                real_list, before_list, after_list_local, output_cols,
                Path(run_dir)/'figs'/f'simulation_compare_epoch{epoch}.png')

        sepp = SEPPTrainer(
                model,
                df_raw,
                groups,
                input_groups,
                output_groups,
                seq_len=seq_len,
                pred_len=pred_len,
                h_max=h_max,
                stride=stride,
                device=device,
                val_loader=sepp_val_loader,
                patience=cfg.get('retrain', {}).get('patience', 5),
                ckpt_path=sepp_ckpt,
                on_improve=on_improve_cb)

        sepp_epochs = cli_args.epochs or cfg.get('retrain', {}).get('epochs', 3)
        sepp.fit(epochs=sepp_epochs)

        trained_model = model
    else:
        # legacy v1/v2 fine-tune path
        retrainer = Retrainer(model_cls=build_model_func,
                              checkpoint=cfg["ckpt"],
                              device=device)

        train_losses, val_losses = retrainer.fine_tune(train_loader, val_loader, epochs=cfg.get("epochs", 3))

        save_loss_plot(train_losses, val_losses, Path(run_dir)/"figs"/"retrain_loss.png", title="Retrain loss")

        new_ckpt = run_dir/"checkpoint_retrained.pth"
        retrainer.model.cpu()
        torch.save(retrainer.model.state_dict(), new_ckpt)
        logger.info(f"Saved fine-tuned checkpoint to {new_ckpt}")

        trained_model = retrainer.model

    # ------------- common post-simulation comparison -------------
    if df_raw is not None:
        from timesim.utils.simulation import simulate_autoregressive
        from timesim.utils.plotting import compare_simulation_plot

        sim_points = cfg.get('retrain', {}).get('sim_points', 1)
        sim_horizon = cfg.get('retrain', {}).get('sim_horizon', 10*pred_len)

        after_list = []
        for sidx in starts_used:
            real, pred, _ = simulate_autoregressive(trained_model,
                                                    df_raw,
                                                    groups,
                                                    input_groups,
                                                    output_groups,
                                                    seq_len=seq_len,
                                                    horizon=sim_horizon,
                                                    device=device,
                                                    start_idx=sidx,
                                                    scaler=scaler,
                                                    use_symlog=bool((cfg.get("data", {}).get("symlog", {}) or {}).get("enabled", False)),
                                                    symlog_columns=(cfg.get("data", {}).get("symlog", {}) or {}).get("columns", None),
                                                    )
            after_list.append(pred)

        multi_compare_simulation_plot(real_list, before_list, after_list, output_cols, Path(run_dir)/'figs'/f'simulation_compare.png')

        np.save(run_dir/'starts.npy', np.array(starts_used))
        np.save(run_dir/'pre_real.npy', np.array(real_list))
        np.save(run_dir/'pre_pred.npy', np.array(before_list))
        np.save(run_dir/'post_pred.npy', np.array(after_list))


if __name__ == "__main__":
    main() 

import hydra
from flatten_dict import flatten
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import gc
gc.collect()
torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg: DictConfig, Console=None) -> None:
    import muon as mu
    import numpy as np
    import wandb
    from LONMF.model import LONMF_3mod

    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project="LONMF_3mod",
        config=flatten(OmegaConf.to_container(cfg), reducer="dot"),
        mode="offline",
    )

    # Load the data.
    mdata = mu.read_h5mu(cfg.data_path + cfg.dataset.dataset_path)

    # Define the Mowgli model.
    model = LONMF_3mod(
        eps=cfg.eps,
        latent_dim=cfg.latent_dim,
        w_regularization=cfg.w_regularization,
        seed=cfg.seed,
        laplace_regularization=cfg.laplace_regularization,
        h_regularization={
             "rna": cfg.h_regularization_rna,
             "adt": cfg.h_regularization_adt,
             "atac": cfg.h_regularization_atac,
        },
    )

    # Fit the Mowgli model.
    model.train(
        mdata,
        max_iter_inner=10_000,
        max_iter=100,
        device="cuda",
        dtype=torch.double,
        lr=1,
        tol_inner=1e-12,
        tol_outer=1e-4,
        optim_name="lbfgs",
        normalize_rows=False,
    )

    for i, l in enumerate(model.losses):
        wandb.log({"outer_iteration": i, "loss": l})
    for i, l in enumerate(model.losses_h):
        wandb.log({"inner_iteration": i, "loss_h": l})
    for i, l in enumerate(model.losses_w):
        wandb.log({"inner_iteration": i, "loss_w": l})

    # 设置CUDA环境变量以减少内存碎片化（可选）
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Save to disk.
    xp_name = f"LONMF_{cfg.latent_dim}_{cfg.eps}"
    xp_name = f"{xp_name}_rna_{cfg.h_regularization_rna}"
    xp_name = f"{xp_name}_adt_{cfg.h_regularization_adt}"
    xp_name = f"{xp_name}_atac_{cfg.h_regularization_atac}"
    xp_name = f"{xp_name}_{cfg.w_regularization}"
    xp_name = f"{xp_name}_{cfg.seed}"
    xp_name = f"{xp_name}_{cfg.laplace_regularization}"
    xp_name = xp_name.replace(".", "_")
    np.save(
        f"{cfg.data_path}{cfg.dataset.save_to_prefix}_{xp_name}.npy",
        {
            "W": mdata.obsm["W_OT"],
            **{"H_" + mod: mdata[mod].uns["H_OT"] for mod in mdata.mod},
        },
    )


if __name__ == "__main__":
    my_app()

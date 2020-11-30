train_config = {
    "seed": 42,

    "data_dir": "/mnt/storage1024/datasets/cassava",

    "train_split": 80,
    "batch_size": 32,
    "num_workers": 0,

    "genet_checkpoint": "./GENet_params/",
    "freeze_percent": 0,
    "loss": "ce",
    "lr": 1e-3,

    "num_epoch": 100,
    "iteration_per_epoch": None,
}

finetune_conf = {
    **train_config,
    **{
        "finetune_version": "version_7",
        "lr": 1e-4,
        "num_epoch": 5,
    }
}
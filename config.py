train_config = {
    "seed": 42,

    "data_dir": "/mnt/storage1024/datasets/cassava",

    "input_shape": (512, 512),
    "train_split": 80,
    "batch_size": 12,
    "num_workers": 0,
    "tta": 3,

    "model_arch": "tf_efficientnet_b3_ns",
    "genet_checkpoint": "./GENet_params/",
    "freeze_percent": 0,
    "loss": "lsce",
    "smoothing": 0.1,
    "lr": 1e-4,
    "adversarial_attack": False,

    "num_epoch": 30,
    "iteration_per_epoch": None,
}

submit_config = {
    "model_arch": train_config["model_arch"],
    "input_shape": train_config["input_shape"],
    "tta": train_config["tta"],
    "data_dir": "/mnt/storage1024/datasets/cassava/test_images",
    "version": "version_0"
}
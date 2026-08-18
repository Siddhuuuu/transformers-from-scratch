from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "gradient_accumulation_steps": 4,

        "nums_epochs": 3,

        "lr": 3e-4,
        "weight_decay": 1e-2,

        "seq_len": 256,

        "d_model": 512,
        "N": 6,
        "h": 8,
        "d_ff": 2048,
        "dropout": 0.1,

        "lang_src": "en",
        "lang_tgt": "hi_en",

        "model_folder": "weights",
        "model_basename": "tmodel_",

        "preload": None,

        "tokenizer_file": "tokenizer_{0}.json",

        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path(".") / model_folder / model_filename)
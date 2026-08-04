from models.unet import UNet
from models.dit import DiT

def build_model(config):
    """Instantiates the model based on the configuration dictionary."""
    model_config = config["model"]
    
    if model_config["type"] == "dit":
        return DiT(
            hidden_dim=model_config["hidden_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            patch_size=model_config["patch_size"],
            in_channels=model_config["in_channels"],
            image_size=model_config["image_size"],
            num_classes=model_config["num_classes"],
            w_min=model_config.get("w_min", 1.0),
            w_max=model_config.get("w_max", 5.0)
        )
    elif model_config["type"] == "unet":
        return UNet(
            time_in=model_config["time_in"],
            time_out=model_config["time_out"],
            down_in_1=model_config["down_in_1"],
            down_in_2=model_config["down_in_2"],
            down_out_1=model_config["down_out_1"],
            down_out_2=model_config["down_out_2"],
            prefinal=model_config["prefinal"],
            num_classes=model_config["num_classes"],
            w_min=model_config.get("w_min", 1.0),
            w_max=model_config.get("w_max", 5.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
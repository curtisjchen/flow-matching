from models.unet import UNet
from models.dit import DiT
import copy

def build_model(config):
    """Instantiates the model based on the configuration dictionary."""
    model_config = config["model"]
    model_type = model_config["type"]

    if model_type == "dit":
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
    if model_type == "unet":
        # Explicitly map the config keys to the new UNet signature
        return UNet(
            w_min=model_config.get("w_min", 1.0),
            w_max=model_config.get("w_max", 5.0),
            in_channels=model_config.get("in_channels", 1), 
            channels=model_config.get("channels", [64, 256]),
            prefinal=model_config.get("prefinal", 32),
            time_in=model_config.get("time_in", 128),
            time_out=model_config.get("time_out", 256),
            num_classes=model_config.get("num_classes", 10)
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # Create a deep copy for the shadow weights
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval() # EMA model always stays in eval mode
        
        # Turn off gradients for the shadow model to save memory
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def step(self, active_model):
        """Call this right after optimizer.step()"""
        with torch.no_grad():
            # If using DDP, we need to access the underlying model weights using .module
            active_params = active_model.module.parameters() if hasattr(active_model, 'module') else active_model.parameters()
            
            for ema_param, active_param in zip(self.ema_model.parameters(), active_params):
                # EMA update formula
                ema_param.data.mul_(self.decay).add_(active_param.data, alpha=1.0 - self.decay)
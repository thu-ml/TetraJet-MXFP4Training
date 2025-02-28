class QuantizationConfig:
    def __init__(self):
        self.qlinear_config = {
            "mlp_fc1": {"all", "linear", "mlp_fc1"},
            "mlp_fc2": {"all", "linear", "mlp_fc2"},
            "attn_proj": {"all", "linear", "attn_proj"},
            "attn_qkv": {"all", "linear", "attn_qkv"},
        }


qconfig = QuantizationConfig()
from omegaconf import OmegaConf

def default_setup(config):
    cfg = OmegaConf.load(config)
    task = cfg.PROJECT.task
    model_name = f"{task}_"
    model_name += cfg.PROJECT.model_idx
    model_info = dict(
        MODEL_INFO=dict(
            model_name=model_name,
        )
    )
    cfg = OmegaConf.merge(cfg, model_info)
    return cfg

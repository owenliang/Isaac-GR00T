# 【中文】模型注册表：全局字典，用于存储模型配置类到Pipeline类的映射
# 键：模型配置类（如 Gr00tN1d6Config）
# 值：Pipeline类（如 Gr00tN1d6Pipeline）
MODEL_REGISTRY = {}


def register_model(model_cfg_cls, pipeline_cls):
    """【中文】注册模型：将模型配置类和对应的Pipeline类注册到全局注册表。
    
    Args:
        model_cfg_cls: 模型配置类（如 Gr00tN1d6Config）
        pipeline_cls: 对应的Pipeline类（如 Gr00tN1d6Pipeline）
    
    Raises:
        ValueError: 如果模型类型已经注册
    """
    if model_cfg_cls in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_cfg_cls}' already registered.")
    MODEL_REGISTRY[model_cfg_cls] = pipeline_cls

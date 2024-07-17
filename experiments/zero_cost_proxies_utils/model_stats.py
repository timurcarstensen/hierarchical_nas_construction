import tensorwatch as tw


def get_model_stats(model, input_tensor_shape, clone_model=True) -> tw.ModelStats:
    # model stats is doing some hooks so do it last
    return tw.ModelStats(model, input_tensor_shape, clone_model=clone_model)

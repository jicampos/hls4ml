import numpy as np
from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.keras_to_hls import parse_default_keras_layer


@pytorch_handler('Flatten')
def parse_flatten_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert pytorch_layer.__class__.__name__ == 'Flatten'

    layer = {}

    layer['name'] = layer_name
    layer['class_name'] = 'Reshape'
    # layer['target_shape'] = [input_shapes[0][0], np.prod(input_shapes[0][1:])]
    layer['target_shape'] = [1, np.prod(input_shapes[0][0:])]
    output_shape = layer['target_shape']

    return layer, output_shape
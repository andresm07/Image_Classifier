import tensorflow.compat.v1 as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from keras.models import load_model
import keras.backend as K


def freeze_graph(session, keep_variable_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_variable_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_variable_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(sess=session, input_graph_def=input_graph_def,
                                                       output_node_names=output_names,
                                                       variable_names_whitelist=freeze_variable_names)
    return frozen_graph


model = load_model("ImageClassifierModel.h5")
K.set_learning_phase(1)
frozen_graph = freeze_graph(K.get_session(), output_names=[model.output.op.name])

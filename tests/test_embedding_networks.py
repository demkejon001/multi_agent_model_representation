import itertools
import torch

from tommas.agent_modellers.embedding_networks import AgentEmbeddingNetwork, lstm_types, pool_types


def test_recurrent_embedding_network():
    row, col = 11, 21
    in_channels = 3
    batch_size = 2
    seq_len = 5
    example_image_batch = torch.randn((seq_len, batch_size, in_channels, row, col))

    embedding_size = 8
    hidden_input_size = 4
    hidden_output_size = 2
    resnet_channels = 4

    flatten_embedding_test_values = [True, False]
    num_resnet_blocks_test_values = [1, 2]
    pre_resnet_layer_test_values = [None, (8, 3, 2, 1), (8, 3, 1, 1), (8, 5, 1, 0)]

    test_values = (flatten_embedding_test_values, num_resnet_blocks_test_values, lstm_types, pool_types,
                   pre_resnet_layer_test_values)
    for (flatten_embedding, num_resnet_blocks, lstm_type, pool_type, pre_resnet_layer) in itertools.product(*test_values):
        net = AgentEmbeddingNetwork(in_channels, (row, col), embedding_size, flatten_embedding=flatten_embedding,
                                    num_resnet_blocks=num_resnet_blocks, resnet_channels=resnet_channels,
                                    recurrent_hidden_size=hidden_output_size, lstm_type=lstm_type, pooling=pool_type,
                                    pre_resnet_layer=pre_resnet_layer)
        output = net(example_image_batch)
        if flatten_embedding:
            assert tuple(output.shape) == (batch_size, embedding_size)
        else:
            if lstm_type == "conv_lstm":
                modified_dim = (row, col)
                if pre_resnet_layer == (8, 3, 2, 1):
                    modified_dim = ((row + 1) // 2, (col + 1) // 2)
                if pre_resnet_layer == (8, 5, 1, 0):
                    modified_dim = (row - 4, col - 4)
                if pool_type is None:
                    assert tuple(output.shape) == (batch_size, embedding_size, *modified_dim)
                else:
                    assert tuple(output.shape) == (batch_size, embedding_size,
                                                   modified_dim[0] // 2, modified_dim[1] // 2)
            else:
                # lstm unflattened is always upsampled to world_dim (row, col)
                assert tuple(output.shape) == (batch_size, embedding_size, row, col)


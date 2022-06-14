import pytest

from tommas.helper_code.conv import make_resnet, torch


def test_make_resnet():
    def assert_correct_resnet_output_shape():
        output = resnet(example_image_batch)
        assert output.shape == (batch_size, out_channels, row, col)

    row, col = 11, 21
    in_channels = 3
    out_channels = 4
    batch_size = 2
    example_image_batch = torch.randn((batch_size, in_channels, row, col))
    # Error: 0 blocks
    with pytest.raises(ValueError):
        make_resnet(in_channels, num_resnet_blocks=0, resnet_channels=0)

    # Error: too many hidden_channels
    with pytest.raises(ValueError):
        make_resnet(in_channels, num_resnet_blocks=2, resnet_channels=[3, 3, 3])

    resnet = make_resnet(in_channels, num_resnet_blocks=1, resnet_channels=out_channels)
    assert_correct_resnet_output_shape()

    # 2 blocks, hidden_channel = [8, out_channels] using int
    resnet = make_resnet(in_channels, num_resnet_blocks=2, resnet_channels=out_channels)
    assert_correct_resnet_output_shape()

    # 2 blocks, hidden_channel = [8, out_channels] using [out_channels]
    resnet = make_resnet(in_channels, num_resnet_blocks=2, resnet_channels=[out_channels])
    assert_correct_resnet_output_shape()

    # 3 blocks, hidden_channel = [32, 8, 8, out_channels] using [32, out_channels]
    resnet = make_resnet(in_channels, num_resnet_blocks=4, resnet_channels=[32, out_channels])
    assert_correct_resnet_output_shape()

    # 3 blocks, hidden_channel = [32, 8, 8, out_channels], using [32, 8, out_channels]
    resnet = make_resnet(in_channels, num_resnet_blocks=4, resnet_channels=[32, 8, out_channels])
    assert_correct_resnet_output_shape()

    # 3 blocks, hidden_channel = [32, 8, 8, out_channels]
    resnet = make_resnet(in_channels, num_resnet_blocks=4, resnet_channels=[32, 8, 8, out_channels])
    assert_correct_resnet_output_shape()



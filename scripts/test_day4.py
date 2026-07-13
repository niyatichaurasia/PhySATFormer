"""Day 4 integration test for PhySATFormer."""

import torch

from src.models.physatformer import PhySATFormer


def main() -> None:
    print("=" * 60)
    print("PhySATFormer - Day 4 Integration Test")
    print("=" * 60)

    batch_size = 8
    sequence_length = 128
    num_channels = 76

    channel_embedding_dim = 32
    d_model = 128
    num_heads = 8
    num_channel_layers = 2
    num_temporal_layers = 4
    ff_dim = 256
    num_classes = 2

    # Dummy physics relationship matrix
    physics_matrix = torch.eye(num_channels)

    model = PhySATFormer(
        input_dim=1,
        channel_embedding_dim=channel_embedding_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_channel_layers=num_channel_layers,
        num_temporal_layers=num_temporal_layers,
        ff_dim=ff_dim,
        num_classes=num_classes,
        physics_matrix=physics_matrix,
        dropout=0.1,
    )

    # Dummy telemetry
    x = torch.randn(
        batch_size,
        sequence_length,
        num_channels,
    )

    print("\nForward Pass")
    print(f"Input Shape : {x.shape}")

    output = model(x)

    print(f"Output Shape: {output.shape}")

    print("\nOutput Sample")
    print(output[0])

    assert output.shape == (
        batch_size,
        num_classes,
    ), (
        f"Expected output shape {(batch_size, num_classes)}, "
        f"got {tuple(output.shape)}."
    )

    print("\nPASS")

    print("\n" + "=" * 60)
    print("DAY 4 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
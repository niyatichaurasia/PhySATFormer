"""Day 3 integration test."""

import torch

from src.models.baseline_transformer import BaselineTransformer


def main() -> None:

    print("=" * 60)
    print("PhySATFormer - Day 3 Integration Test")
    print("=" * 60)

    model = BaselineTransformer(
        input_dim=3,
        d_model=128,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        num_classes=2,
        dropout=0.1,
    )

    x = torch.randn(
        8,
        128,
        3,
    )

    logits = model(x)

    print("\nForward Pass")

    print(f"Input Shape : {x.shape}")
    print(f"Output Shape: {logits.shape}")

    assert logits.shape == (8, 2)

    print("\nOutput Sample")

    print(logits[0])

    print("\nPASS")

    print("\n" + "=" * 60)
    print("DAY 3 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""Run all integration tests."""

from scripts.test_day1 import main as day1
from scripts.test_day2 import main as day2
from scripts.test_day3 import main as day3


def main() -> None:

    print("=" * 80)
    print("PhySATFormer Integration Test Suite")
    print("=" * 80)

    print("\nRunning Day 1...")
    day1()

    print("\nRunning Day 2...")
    day2()

    print("\nRunning Day 3...")
    day3()

    print("\n" + "=" * 80)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
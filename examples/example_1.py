import argparse

import GenerativeAnatomy


def main(number=10):
    print(number)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example example")
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        nargs="+",
        default=10,
        help="Number of times to call the example function",
    )
    args = parser.parse_args()
    print(args)

    main(**vars(args))

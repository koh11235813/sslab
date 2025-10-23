import torch


def main() -> None:
    print("torch", torch.__version__, "cuda?", torch.cuda.is_available())


if __name__ == "__main__":
    main()

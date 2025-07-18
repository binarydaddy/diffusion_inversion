from research_experiments import collect_data_wrapper

def main(args):
    collect_data_wrapper(args.caption_path, args.seed, args.device)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--caption_path", type=str, default="captions.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
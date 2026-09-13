import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="OLAP lake service")
    parser.add_argument("--load_config", type=str)
    parser.add_argument("--web_host", type=str)
    parser.add_argument("--web_port", type=int)
    parser.add_argument("--sqlite_path", type=str)
    return parser.parse_known_args()

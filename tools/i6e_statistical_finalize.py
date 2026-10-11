"""Automatically publish all 312 dispositions, preserving incomplete scientific arms."""

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

from tools.i6e_statistical_campaign import status
from tools.i6e_statistical_dispatch import close_dispositions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--url",required=True)
    p.add_argument("--token-file",type=Path,required=True)
    p.add_argument("--code-commit",required=True)
    args=p.parse_args()
    lock=(args.output/"COORDINATOR.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (args.output/"olap/PUBLICATION.json").exists():
        print("already published")
        return
    current=status(args.output)
    if current["completed"]+current["failed"]<312:
        print(json.dumps(current))
        return
    close_dispositions(args.output)
    subprocess.run([sys.executable,"-m","tools.i6e_publish_olap","--statistical-root",str(args.output),
        "--allow-dispositions","--output",str(args.output/"olap"),"--url",args.url,
        "--token-file",str(args.token_file),"--code-commit",args.code_commit],check=True)


if __name__=="__main__":
    main()

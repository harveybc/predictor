"""Restore a digest-checked request inside an admitted scope, then exec its child.

Some launchers start scopes asynchronously and disconnect stdin. The supervisor
therefore supplies a private request file; this helper runs after admission.
"""
import hashlib
import os
import sys


def main():
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("missing child command")
    path = os.environ.pop("I6N_SWARM_REQUEST_FILE", None)
    digest = os.environ.pop("I6N_SWARM_REQUEST_SHA256", None)
    if path is not None:
        with open(path, "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("request digest mismatch")
            stream.seek(0)
            os.dup2(stream.fileno(), 0, inheritable=True)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()

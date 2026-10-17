"""Presentation-only close operator over the frozen original scientific runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time


def render(cells, results):
    lines=["# Continuation Results","","DEVELOPMENT 2024; TEST closed. Negative relative error is better.","",
           "| Cell | State | Parameters | Fit Seconds | Per-Horizon Relative Error |",
           "|---|---|---:|---:|---|"]
    for cell in cells:
        r=results.get(cell["id"],{});values=[]
        for row in r.get("metrics",{}).get("VALIDATION",[]):
            value=row["relative_error"]
            values.append(f"h{row['horizon']}: UNAVAILABLE (n={row['n']})" if value is None
                          else f"h{row['horizon']}: {100*value:+.6f}%")
        lines.append(f"| {cell['id']} | {r.get('state','PENDING')} | {r.get('parameters','')} | "
                     f"{r.get('training_seconds','')} | {'; '.join(values)} |")
    lines+=["","Unsupported heads remain unavailable; no metric, mask, model or baseline was changed."]
    return "\n".join(lines)+"\n"


def checksum_chunks(files):
    paths=sorted(files)
    return [paths[i:i+64] for i in range(0,len(paths),64)]


def verify_checksums(expected, output):
    actual={}
    for line in output.splitlines():
        digest,separator,path=line.partition("  ")
        if separator:actual[path]=digest
    if actual!=expected:raise ValueError("BATCH_MIRROR_MISMATCH")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--spec",required=True);args=parser.parse_args()
    spec=json.loads(Path(args.spec).read_text())
    from tools.i6dmo import resolution_batch as B
    expected=Path(spec["runtime"]).resolve()
    if not Path(B.__file__).resolve().is_relative_to(expected):
        raise ValueError("FROZEN_SCIENTIFIC_RUNTIME_REQUIRED")
    def summarize_only(s, root):
        root=Path(root)
        results={c["id"]:r for c in s["cells"] if (r:=B.terminal(root,c,s))}
        tmp=root/"RESULTS.md.tmp";tmp.write_text(render(s["cells"],results));tmp.replace(root/"RESULTS.md")
    original_mirror=B.mirror_tree
    def bounded_mirror(root, destination, role=None):
        if not role or not role.get("ssh_alias"):return original_mirror(root,destination,role)
        root,destination=Path(root),Path(destination)
        files={str(p.relative_to(root)):B.file_sha256(p) for p in root.rglob("*")
               if p.is_file() and p.name!="mirror.lock"}
        B.command(["ssh",role["ssh_alias"],shlex.join(["mkdir","-p",str(destination)])])
        B.command(["rsync","-a","--checksum",str(root)+"/",role["ssh_alias"]+":"+str(destination)+"/"],600)
        for paths in checksum_chunks(files):
            script="cd "+shlex.quote(str(destination))+" && sha256sum -- "+" ".join(shlex.quote(p) for p in paths)
            output=B.command(["ssh",role["ssh_alias"],script],120)
            verify_checksums({p:files[p] for p in paths},output)
        return dict(state="VERIFIED",count=len(files),manifest_sha256=B.digest(files))
    # Only Markdown null handling and checksum transport differ; scientific facts and gates stay original.
    B.summarize=summarize_only
    B.mirror_tree=bounded_mirror
    proof=dict(schema="i6dmo.presentation_close_operator.v1",at=time.time(),scientific_code_commit=spec["code_commit"],
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        substituted_hook="summarize_null_support_and_bounded_remote_checksum_transport",fit_executed=False,test_read=False)
    if not (Path(spec["root"])/"CLOSE.json").exists():
        B.put(Path(spec["artifacts"])/"CLOSURE_OPERATOR.json",proof)
    status=B.tick(spec)
    print(json.dumps({k:status.get(k) for k in ("state","complete","terminal","total","failed","mirrors")},sort_keys=True))


if __name__=="__main__":main()

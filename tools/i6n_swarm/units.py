"""Durable systemd *user* units for ONE campaign (install never touches any other unit).

Every unit file carries ``X-I6N-Swarm-Campaign=<id>`` in [Unit]; install
refuses to overwrite a file of the same name without that marker. Units:

coordinator: <p>-ledger.service (lease service, Restart=on-failure),
             <p>-tick.service + <p>-tick.timer (DAG activation + status).
worker role: <p>-<role>-node.service (DOIN node, Restart=on-failure),
             <p>-<role>-agent.service + <p>-<role>-agent.timer
             (boot recovery, healthcheck, fixed units, node-config sync).
"""
from pathlib import Path
import shlex

from tools.i6n_swarm import plan as P

MARK = "X-I6N-Swarm-Campaign"


def prefix(campaign_id):
    return f"i6n-swarm-{campaign_id}-"


def startable_units(campaign_id, role):
    p = prefix(campaign_id)
    if role == "coordinator":
        return [f"{p}ledger.service", f"{p}tick.timer"]
    return [f"{p}{role}-node.service", f"{p}{role}-agent.timer"]


def _q(args):
    return " ".join(shlex.quote(str(a)) for a in args)


def render(root, role):
    root = Path(root)
    payload = P.load(root)["payload"]
    private = P.private(root)
    cid = payload["campaign_id"]
    p = prefix(cid)
    host = private["hosts"][role]
    python, workdir = host["python"], host["workdir"]
    head = (f"[Unit]\nDescription=I6-N swarm {cid} {{what}}\n{MARK}={cid}\n"
            "After=network-online.target\n")
    env = f"Environment=PYTHONDONTWRITEBYTECODE=1\nWorkingDirectory={workdir}\n"
    units = {}
    if role == "coordinator":
        ledger = private["ledger"]
        units[f"{p}ledger.service"] = (
            head.format(what="ledger lease service") + "\n[Service]\nType=simple\n" + env +
            "ExecStart=" + _q([python, "-m", "tools.i6n_swarm.lease_service", "--ledger",
                               root / "state" / "ledger.sqlite3", "--bind", ledger["bind"], "--port", ledger["port"],
                               "--token-file", ledger["token_file"]]) +
            "\nRestart=on-failure\nRestartSec=10\nMemoryMax=512M\n\n[Install]\nWantedBy=default.target\n")
        units[f"{p}tick.service"] = (
            head.format(what="DAG tick (activation, recovery, status)") + "\n[Service]\nType=oneshot\n" + env +
            "Environment=CUDA_VISIBLE_DEVICES=\nExecStart=" +
            _q([python, "-m", "tools.i6n_swarm_campaign", "tick", "--root", root]) + "\nMemoryMax=1G\nTimeoutStartSec=900\n")
        units[f"{p}tick.timer"] = (
            head.format(what="DAG tick timer") + "\n[Timer]\nOnBootSec=2min\n"
            f"OnUnitInactiveSec={int(private.get('tick_interval_seconds', 300))}s\nPersistent=true\n"
            f"Unit={p}tick.service\n\n[Install]\nWantedBy=timers.target\n")
        return units
    node_cfg = root / "private" / "hosts" / role / "node.json"
    doin_cfg = root / "private" / "hosts" / role / "doin_node.json"
    node_env = env + f"Environment=I6N_SWARM_NODE_CONFIG={node_cfg}\n"
    for key, value in sorted((host.get("environment") or {}).items()):
        node_env += f"Environment={key}={value}\n"
    units[f"{p}{role}-node.service"] = (
        head.format(what=f"{role} DOIN node (shared swarm)") + f"ConditionPathExists={doin_cfg}\n\n[Service]\n"
        "Type=simple\n" + node_env +
        "ExecStartPre=" + _q([python, "-m", "tools.i6n_swarm.agent", "--node-config", node_cfg, "--recover-only"]) +
        "\nExecStart=" + _q([*private["doin"]["node_command"], "--config", doin_cfg]) +
        "\nRestart=on-failure\nRestartSec=30\nTimeoutStopSec=120\nKillMode=mixed\n\n[Install]\nWantedBy=default.target\n")
    units[f"{p}{role}-agent.service"] = (
        head.format(what=f"{role} agent (recovery, healthcheck, fixed units, node sync)") +
        "\n[Service]\nType=oneshot\n" + node_env +
        "ExecStart=" + _q([python, "-m", "tools.i6n_swarm.agent", "--node-config", node_cfg]) +
        "\nRestart=on-failure\nRestartSec=60\nTimeoutStartSec=infinity\nKillMode=mixed\n")
    units[f"{p}{role}-agent.timer"] = (
        head.format(what=f"{role} agent timer") + "\n[Timer]\nOnBootSec=3min\nOnUnitInactiveSec=120s\nPersistent=true\n"
        f"Unit={p}{role}-agent.service\n\n[Install]\nWantedBy=timers.target\n")
    return units


def install(root, roles, unit_dir=None, systemctl=None):
    from tools.i6n_swarm.coordinator import default_systemctl
    payload = P.load(root)["payload"]
    cid = payload["campaign_id"]
    unit_dir = Path(unit_dir or Path.home() / ".config" / "systemd" / "user")
    unit_dir.mkdir(parents=True, exist_ok=True)
    systemctl = systemctl or default_systemctl
    rendered = {}
    for role in roles:
        rendered.update(render(root, role))
    for name in rendered:
        if not name.startswith(prefix(cid)):
            raise ValueError(f"refusing to write non-campaign unit {name}")
        path = unit_dir / name
        if path.exists() and f"{MARK}={cid}" not in path.read_text():
            raise ValueError(f"refusing to overwrite foreign unit file {name}")
    written, unchanged = [], []
    for name, text in sorted(rendered.items()):
        path = unit_dir / name
        if path.exists() and path.read_text() == text:
            unchanged.append(name)
            continue
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)
        written.append(name)
    if written:
        systemctl("daemon-reload")
    enable = [n for n in sorted(rendered) if "[Install]" in rendered[n]]
    if enable:
        systemctl("enable", *enable)
    return dict(units=sorted(rendered), written=written, unchanged=unchanged, enabled=enable)

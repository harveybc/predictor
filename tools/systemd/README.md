# Managed governance forwarding

`crispdm-governance-tunnel@.service` keeps worker-local loopback ports 15055 and
15057 connected to the coordinator's existing 5055 and 5057 services. The
instance parameter is an existing SSH host alias, not a credential or public
address. Install on the coordinator, whose SSH configuration already reaches
that worker. It does not start/restart a store or open a public listener.

Before adoption, inspect both worker ports and active experiments. Do not replace
an incumbent forwarding process blindly. Verify the coordinator services first.
After the alias is configured, install the template in the user's systemd unit
directory, run `systemctl --user daemon-reload`, then enable/start the selected
instance. Existing user-manager persistence/linger must be checked separately;
this template does not configure it.

Example with a placeholder alias:

```bash
systemd-analyze --user verify tools/systemd/crispdm-governance-tunnel@.service
systemctl --user enable --now crispdm-governance-tunnel@<worker-alias>.service
systemctl --user show crispdm-governance-tunnel@<worker-alias>.service \
  -p ActiveState -p SubState -p NRestarts
```

The enable command assumes the template has already been installed. Server
keepalive detects a lost SSH connection and systemd reconnects; it cannot keep
the coordinator reachable while that machine is off. `ExitOnForwardFailure`
refuses a conflicting listener rather than pretending forwarding succeeded.

After recovery, perform the existing authenticated governed operation and content
reconciliation. A root-page HTTP200/302 or a listening port proves transport, not
authorization, terminal acceptance or scientific validity. Retry an existing
terminal from its canonical envelope; do not rebuild timestamps or retrain a
completed cell to repair transport. Preserve rejected/conflicting envelopes.

Adoption evidence: RP135 resumption, 2026-09-23. Both user instances active with
NRestarts=0; remote HTTP responses restored; recovered second-cell terminal
accepted with empty reconciliation. Unit validation emitted an unrelated warning
from the system's existing spice-vdagent unit. No store service was restarted.

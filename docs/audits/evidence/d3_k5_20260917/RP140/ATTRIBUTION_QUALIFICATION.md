# Reading the two original-device replay evidence files

`docs/audits/evidence/d3_k5_20260917/RP138/ORIGINAL_DEVICE_REPLAY_WORKER_A.json` and
`ORIGINAL_DEVICE_REPLAY_COORDINATOR.json` are preserved unchanged. Their `device` field is the device the **replay** ran on,
measured by UUID. It is not a statement about which physical device trained the cell.

The RP140 composition derives that distinction from the records themselves. Over the twelve protocol-A cells,
`device_attribution` returns UNKNOWN for four and INFERRED_GPU_MEMORY for eight, and MEASURED for none. Every cell with a
bound exact replay is therefore classed `REPLAY_EXACT_ON_OBSERVED_DEVICE`, never `SAME_DEVICE_REPLAY_MEASURED`.

What the replays establish: the stored predictions are reproduced element for element from the retained checkpoint on the
observed device, under the frozen rule. What they do not establish: the physical identity of the training device.

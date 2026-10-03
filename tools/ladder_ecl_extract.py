"""One-time extraction of windows/targets from the ECL train/validation NPZ to .npy (memory-mappable). Values are bit-identical."""
import sys, os, numpy as np
d, o = sys.argv[1], sys.argv[2]; os.makedirs(o, exist_ok=True)
for s in ("train", "validation"):
    z = np.load(os.path.join(d, s + ".npz"), allow_pickle=False)
    for k in ("windows", "targets"):
        a = z[k]; np.save(os.path.join(o, "%s_%s.npy" % (s, k)), a); print(s, k, a.shape, a.dtype); del a

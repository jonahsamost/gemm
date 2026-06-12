import time
from triton.testing import do_bench


def bench_and_report(name, fn, flops, warmup=5, rep=30, gbps_bytes=0):
    time.sleep(0.5)
    t = do_bench(fn, warmup=warmup, rep=rep)
    tflops = flops / (t * 1e9)
    if gbps_bytes:
        gbps = gbps_bytes / (t * 1e6)
        print(f"{name}: {t:.3f} ms,  {tflops:7.1f} TFLOP/s,  {gbps:.0f} GB/s")
    else:
        print(f"{name}: {t:.3f} ms,  {tflops:7.1f} TFLOP/s")
    return t

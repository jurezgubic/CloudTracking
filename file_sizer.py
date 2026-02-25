# save as nc_var_sizes.py, then: python nc_var_sizes.py cloud_results_lba.nc
import sys
from pathlib import Path

import h5py

def fmt(n: int) -> str:
    n = float(n)
    for u in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024 or u == "TiB":
            return f"{n:,.2f}{u}" if u != "B" else f"{int(n):,}{u}"
        n /= 1024

def main(fn: str, top: int = 30) -> None:
    rows = []
    with h5py.File(fn, "r") as f:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            # skip internal dimension-scale datasets if present
            if obj.attrs.get("CLASS") == b"DIMENSION_SCALE":
                return

            alloc = int(obj.id.get_storage_size())           # on-disk bytes
            logical = int(obj.size * obj.dtype.itemsize)     # dtype * elements
            ratio = (logical / alloc) if alloc else float("inf")

            rows.append((alloc, logical, ratio, name, obj.shape, obj.chunks, obj.compression))

        f.visititems(visit)

    rows.sort(key=lambda x: x[0], reverse=True)

    file_bytes = Path(fn).stat().st_size
    print(f"File: {fn} (total on disk: {fmt(file_bytes)})")
    print(f"{'alloc':>10} {'logical':>10} {'L/A':>7}  {'comp':>10} {'chunks':>18} {'shape':>18}  name")

    for alloc, logical, ratio, name, shape, chunks, comp in rows[:top]:
        print(f"{fmt(alloc):>10} {fmt(logical):>10} {ratio:7.2f}  {str(comp):>10} {str(chunks):>18} {str(shape):>18}  /{name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python nc_var_sizes.py <file.nc> [topN]")
    fn = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) >= 3 else 30
    main(fn, top=top)


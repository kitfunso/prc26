"""Open ADS-B ground traces (adsb.lol globe_history, ODbL 1.0) cut to the ten airports, one parquet per day.

python adsb_fetch.py [N_DL] [N_EX]   downloads every day in DAYS not yet cut, keeps points within RADIUS_KM of an airport
and below ALT_FT above it, writes data/adsb/pts/YYYY-MM-DD.parquet and deletes the raw archive.
"""
import gzip, io, json, os, subprocess, sys, tarfile, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, timedelta
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW, PTS = os.path.join(ROOT, "data", "adsb", "raw"), os.path.join(ROOT, "data", "adsb", "pts")
LOG = os.path.join(ROOT, "data", "adsb", "fetch.log")
APT = {"EDDF": (50.026706, 8.558350, 364), "EDDM": (48.353802, 11.786100, 1487), "EGLL": (51.470748, -0.459909, 83),
       "EHAM": (52.308601, 4.763890, -11), "LEBL": (41.297100, 2.078460, 12), "LEMD": (40.493407, -3.572249, 1998),
       "LFPG": (49.008960, 2.554117, 392), "LIRF": (41.804532, 12.251998, 13), "LSZH": (47.458056, 8.548056, 1417),
       "LTFM": (41.274874, 28.732136, 325)}
CODES = list(APT)
LAT, LON, ELEV = (np.array([APT[c][i] for c in CODES]) for i in range(3))
RADIUS_KM, ALT_FT = 8.0, 3000.0


def months(y, ms):
    return [date(y, m, 1) + timedelta(d) for m in ms for d in range(31) if (date(y, m, 1) + timedelta(d)).month == m]


# Four labelled days first to build the detector on; the 2025 holdout months come before July 2026 so the gate can run
# while the rest of the board months download. Late 2025 leads the extra training months: its coverage is nearest 2026's.
DEV = [date(2025, 1, 15), date(2025, 7, 15), date(2025, 1, 16), date(2025, 7, 16)]
DAYS = DEV + [d for d in months(2026, [1]) + months(2025, [1, 7]) + months(2026, [7, 2, 6]) + months(2025, [2, 6])
              + months(2025, [12, 11, 10, 9, 8, 5, 4, 3])
              if d not in DEV]


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def tag(d):
    return f"v{d:%Y.%m.%d}-planes-readsb-prod-0"


def assets(d):
    out = subprocess.run(["gh", "api", f"repos/adsblol/globe_history_{d.year}/releases/tags/{tag(d)}",
                          "--jq", '.assets[] | "\\(.name) \\(.size)"'], capture_output=True, text=True, check=True).stdout
    parts = [(n, int(s)) for n, s in (l.split() for l in out.splitlines()) if ".tar." in n or n.endswith(".tar")]
    if not parts:  # some days ship as one .tar, others split; neither found means no data, not an empty day
        raise RuntimeError(f"{d} has no tar asset")
    return sorted(parts)


def download(d):
    parts = assets(d)
    procs = []
    for name, size in parts:
        p = os.path.join(RAW, name)
        if os.path.exists(p) and os.path.getsize(p) == size:
            continue
        url = f"https://github.com/adsblol/globe_history_{d.year}/releases/download/{tag(d)}/{name}"
        # A stalled connection never errors, so a speed floor turns it into a timeout that --retry resumes.
        procs.append(subprocess.Popen(["curl", "-sfL", "--retry", "20", "--retry-all-errors", "--speed-limit", "200000",
                                       "--speed-time", "45", "-C", "-", "-o", p, url]))
    for pr in procs:
        pr.wait()
    paths = [os.path.join(RAW, n) for n, _ in parts]
    bad = [n for n, s in parts if not os.path.exists(os.path.join(RAW, n)) or os.path.getsize(os.path.join(RAW, n)) != s]
    if bad:
        raise RuntimeError(f"{d} incomplete parts {bad}")
    return paths


class Concat(io.RawIOBase):
    def __init__(self, paths):
        self.paths, self.i, self.f = paths, 0, open(paths[0], "rb")

    def readinto(self, b):
        n = self.f.readinto(b)
        while n == 0:
            self.f.close(); self.i += 1
            if self.i >= len(self.paths):
                return 0
            self.f = open(self.paths[self.i], "rb"); n = self.f.readinto(b)
        return n

    def readable(self):
        return True


def cut(d, paths):
    tf = tarfile.open(fileobj=io.BufferedReader(Concat(paths), 8 << 20), mode="r|")
    cols = {k: [] for k in ("icao", "r", "t", "ts", "lat", "lon", "alt", "gs", "trk", "flags", "src", "cs")}
    n_files = 0
    for m in tf:
        if not (m.isfile() and "trace_full_" in m.name):
            continue
        n_files += 1
        try:
            obj = json.loads(gzip.decompress(tf.extractfile(m).read()))
        except (OSError, ValueError) as e:
            log(f"{d} skip {m.name}: {e}")
            continue
        tr = obj.get("trace") or []
        if not tr:
            continue
        lat = np.array([p[1] if p[1] is not None else np.nan for p in tr], dtype=float)
        lon = np.array([p[2] if p[2] is not None else np.nan for p in tr], dtype=float)
        # Coarse box first: a degree of latitude is 111 km, so 0.1 deg clears RADIUS_KM.
        near = (np.abs(lat[:, None] - LAT) < 0.1) & (np.abs(lon[:, None] - LON) < 0.15)
        cand = np.where(near.any(1))[0]
        if not len(cand):
            continue
        la, lo = np.radians(lat[cand])[:, None], np.radians(lon[cand])[:, None]
        h = np.sin((np.radians(LAT) - la) / 2) ** 2 + np.cos(la) * np.cos(np.radians(LAT)) * np.sin((np.radians(LON) - lo) / 2) ** 2
        km = 12742.0 * np.arcsin(np.sqrt(h))
        j = km.argmin(1)
        ok = km[np.arange(len(cand)), j] <= RADIUS_KM
        cand, j = cand[ok], j[ok]
        if not len(cand):
            continue
        cs, last = [], None
        ci = 0
        keep = []
        for i, p in enumerate(tr):  # callsign rides in the occasional details dict, so carry it forward
            det = p[8] if len(p) > 8 else None
            if isinstance(det, dict) and det.get("flight"):
                last = det["flight"].strip()
            if ci < len(cand) and i == cand[ci]:
                a = p[3]
                alt = np.nan if a == "ground" or a is None else float(a)
                if a == "ground" or (alt == alt and alt - ELEV[j[ci]] < ALT_FT):
                    keep.append((i, alt, last))
                ci += 1
        if not keep:
            continue
        base = float(obj.get("timestamp", 0.0))
        for i, alt, c in keep:
            p = tr[i]
            cols["icao"].append(obj.get("icao", "")); cols["r"].append(obj.get("r")); cols["t"].append(obj.get("t"))
            cols["ts"].append(base + p[0]); cols["lat"].append(lat[i]); cols["lon"].append(lon[i])
            cols["alt"].append(-1e9 if p[3] == "ground" else alt)
            cols["gs"].append(p[4] if len(p) > 4 and p[4] is not None else np.nan)
            cols["trk"].append(p[5] if len(p) > 5 and p[5] is not None else np.nan)
            cols["flags"].append(p[6] if len(p) > 6 and p[6] is not None else 0)
            cols["src"].append(p[9] if len(p) > 9 else None); cols["cs"].append(c)
    df = pd.DataFrame(cols)
    for c in ("lat", "lon", "gs", "trk"):
        df[c] = df[c].astype("float32")
    df["ground"] = df.alt == -1e9
    df["alt"] = df.alt.where(~df.ground).astype("float32")
    df["flags"] = df["flags"].astype("int16")
    for c in ("icao", "r", "t", "src", "cs"):
        df[c] = df[c].astype("category")
    return df, n_files


def extract(d, paths):
    t0 = time.time()
    df, n = cut(d, paths)
    out = os.path.join(PTS, f"{d:%Y-%m-%d}.parquet")
    df.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    for p in paths:
        os.remove(p)
    return f"{d} cut {n} traces to {len(df)} points, {df.icao.nunique()} aircraft, {time.time() - t0:.0f}s"


def main(n_dl=4, n_ex=8):
    os.makedirs(RAW, exist_ok=True); os.makedirs(PTS, exist_ok=True)
    todo = [d for d in DAYS if not os.path.exists(os.path.join(PTS, f"{d:%Y-%m-%d}.parquet"))]
    log(f"start {len(todo)} days, {n_dl} downloads, {n_ex} extractors")
    with ProcessPoolExecutor(n_ex) as ex, ThreadPoolExecutor(n_dl) as dl:
        dls = {d: dl.submit(download, d) for d in todo}
        exs = []
        for d in todo:
            try:
                paths = dls[d].result()
            except Exception as e:
                log(f"{d} download FAILED {e}")
                continue
            log(f"{d} downloaded")
            exs.append((d, ex.submit(extract, d, paths)))
            while sum(not f.done() for _, f in exs) >= 2 * n_ex:  # keep raw days on disk bounded
                time.sleep(5)
        for d, f in exs:
            try:
                log(f.result())
            except Exception as e:
                log(f"{d} extract FAILED {e!r}")
    log("end")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))

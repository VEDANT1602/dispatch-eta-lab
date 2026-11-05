"""Fictional delivery events with a declared time and identity contract."""

import csv
import gzip
import hashlib
import io
import math
import random
from pathlib import Path

import numpy as np


DAY = 1440
FEATURES = ("distance_km", "traffic_index", "load_ratio", "available_couriers", "rain",
            "basket_items", "hour_sin", "hour_cos", "weekend", "zone_core", "zone_hills")
FIELDS = ("event_id", "dispatch_minute", "day", "hour", "route_id", "courier_id", "zone",
          "rush", "shift", "feature_available_minute", *FEATURES, "eta_minutes", "label_available_minute")


def phase(day):
    if day < 18:
        return "train"
    if 20 <= day < 24:
        return "calibration"
    if 26 <= day < 36:
        return "test"
    return "embargo"


def generate(seed=73, events_per_day=180):
    if type(seed) is not int or type(events_per_day) is not int or events_per_day < 10:
        raise ValueError("integer seed and at least ten events per day required")
    rng = random.Random(seed)
    events = []
    for day in range(36):
        cohort = phase(day)
        for index in range(events_per_day):
            minute = day * DAY + (index + 0.5) * DAY / events_per_day
            hour = (minute % DAY) / 60
            rush = int(7 <= hour < 10 or 16 <= hour < 20)
            zone = rng.choice(("core", "suburban", "hills"))
            distance = rng.uniform(0.6, 13)
            rain = int(rng.random() < 0.23)
            traffic = min(2.1, max(0.15, rng.gauss(0.5 + 0.7 * rush + 0.2 * rain, 0.19)))
            available = rng.randint(3, 18)
            queue = rng.randint(1, 40) + 9 * rush
            load = queue / available
            basket = rng.randint(1, 18)
            travel = 2.15 * distance * (1 + 0.46 * traffic ** 1.7) * (1 + 0.14 * rain)
            wait = 3 + 1.4 * max(0, load - 1) ** 1.35 + 3.5 * rush * (zone == "core")
            handoff = 0.55 * basket + 3 * (zone == "hills") + 0.8 * rain * basket
            noise = rng.gauss(0, 1.3 + 0.9 * traffic + 1.6 * (zone == "hills"))
            exceptional_delay = rng.uniform(6, 15) if rng.random() < 0.025 else 0
            shifted = int(day >= 31)
            # Unobserved closure/handling change: diagnostic flag is never a feature.
            disruption = shifted * (8 + 8 * (zone == "hills") + 4 * rush)
            eta = max(1, travel + wait + handoff + noise + exceptional_delay + disruption)
            event = {
                "event_id": f"D{day:02}-{index:04}", "dispatch_minute": round(minute, 6),
                "day": day, "hour": round(hour, 6),
                "route_id": f"{cohort}-route-{rng.randrange(90):03}",
                "courier_id": f"{cohort}-courier-{rng.randrange(120):03}", "zone": zone,
                "rush": rush, "shift": shifted, "feature_available_minute": round(minute - 1, 6),
                "distance_km": round(distance, 6), "traffic_index": round(traffic, 6),
                "load_ratio": round(load, 6), "available_couriers": available, "rain": rain,
                "basket_items": basket, "hour_sin": round(math.sin(2 * math.pi * hour / 24), 6),
                "hour_cos": round(math.cos(2 * math.pi * hour / 24), 6), "weekend": int(day % 7 >= 5),
                "zone_core": int(zone == "core"), "zone_hills": int(zone == "hills"),
                "eta_minutes": round(eta, 6), "label_available_minute": round(minute + eta + rng.uniform(5, 60), 6),
            }
            events.append(event)
    return events


def split(events):
    if not events or len({e["event_id"] for e in events}) != len(events):
        raise ValueError("nonempty events with unique event IDs required")
    for event in events:
        for field in ("dispatch_minute", "feature_available_minute", "eta_minutes", "label_available_minute"):
            value = event.get(field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("finite numeric time/duration required: " + field)
        if type(event.get("day")) is not int or not 0 <= event["day"] < 36:
            raise ValueError("day must be an integer in the declared simulation")
    ordered = sorted(events, key=lambda e: (e["dispatch_minute"], e["event_id"]))
    groups = {name: [e for e in ordered if phase(e["day"]) == name] for name in ("train", "calibration", "test")}
    if any(not rows for rows in groups.values()):
        raise ValueError("all three time partitions must be nonempty")
    for e in ordered:
        if e["day"] != int(e["dispatch_minute"] // DAY):
            raise ValueError("day disagrees with dispatch time")
        if e["feature_available_minute"] > e["dispatch_minute"]:
            raise ValueError("future feature")
        if not math.isfinite(e["eta_minutes"]) or e["eta_minutes"] < 0:
            raise ValueError("nonnegative finite duration required")
        if e["label_available_minute"] < e["dispatch_minute"] + e["eta_minutes"]:
            raise ValueError("label precedes delivery completion")
    for earlier, later, cutoff in (("train", "calibration", 20 * DAY), ("calibration", "test", 26 * DAY)):
        if max(e["label_available_minute"] for e in groups[earlier]) >= cutoff:
            raise ValueError("earlier labels were not available before the next partition")
        for key in ("route_id", "courier_id"):
            if {e[key] for e in groups[earlier]} & {e[key] for e in groups[later]}:
                raise ValueError("identity overlap across partitions")
    for key in ("route_id", "courier_id"):
        if {e[key] for e in groups["train"]} & {e[key] for e in groups["test"]}:
            raise ValueError("identity overlap across partitions")
    return groups


def matrix(events):
    x = np.asarray([[e[name] for name in FEATURES] for e in events], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite feature")
    return x


def targets(events):
    return np.asarray([e["eta_minutes"] for e in events], dtype=np.float64)


def package(events, path):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(events)
    raw = stream.getvalue().encode("utf-8")
    packed = gzip.compress(raw, mtime=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packed)
    return {"rows": len(events), "csv_sha256": hashlib.sha256(raw).hexdigest(),
            "gzip_sha256": hashlib.sha256(packed).hexdigest(), "compressed_bytes": len(packed)}

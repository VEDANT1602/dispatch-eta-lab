# Data card: original fictional delivery events

Created for this independent project by Vedant Thaker on November 5, 2026. The generator, records and labels are original; the [MIT license](../LICENSE) covers them. No third-party dataset, scraped deliveries, real route coordinates, personal courier data or customer records are included.

## Grain and mechanism

One row is one fictional dispatch. The default stream spans 36 logical days with 180 evenly spaced dispatches per day, for 6,480 events. Day zero is an abstract simulation origin, not a real operational date. The generator uses seed 73. Distances, traffic, weather, basket size, available couriers and queue/supply pressure are sampled inputs. Three fictional service zones are `core`, `suburban` and `hills`; they are operational groups, not protected demographic categories.

The duration combines nonlinear travel, queue and handoff components with heteroskedastic noise and occasional exceptional delay. From day 31, the generator adds an unobserved disruption: eight minutes, eight more for hills, and four more in rush hour. This constructed stress condition is public and intentionally unfavorable to a frozen model/interval. It is not evidence about a real company's delays.

## Availability and leakage boundary

| Fields | Availability and use |
| --- | --- |
| `distance_km`, `traffic_index`, `load_ratio`, `available_couriers`, `rain`, `basket_items` | Declared observable at dispatch; input features |
| `hour_sin`, `hour_cos`, `weekend`, `zone_core`, `zone_hills` | Known calendar/zone encoding; input features |
| `event_id`, `route_id`, `courier_id` | Audit identities; excluded from model inputs |
| `dispatch_minute`, `day`, `hour`, `zone`, `rush` | Timing/group audit fields; absolute time and raw groups are not model columns |
| `shift` | Diagnostic mechanism label, unavailable to a deployed predictor; excluded |
| `feature_available_minute` | Contract check: no feature snapshot may arrive after dispatch |
| `eta_minutes` | Target; unavailable until delivery completion |
| `label_available_minute` | Completion plus reporting delay; used to validate label maturity, never as a predictor |

Route and courier IDs are generated from separate train/calibration/test cohorts. The model cannot memorize them; tests also reject accidental identity overlap. The independent cohort names do not recreate an actual geographic or courier cold-start distribution. The generator does not use learned historical aggregates, GPS routes, a routing engine or a live supply service.

## Time partitions

Training uses days `[0,18)` (3,240 rows); calibration `[20,24)` (720); test `[26,36)` (1,800). Days 18–19 and 24–25 are excluded (720). Earlier labels must be available before the next partition starts. The two test periods each have 900 deliveries: before the shift and during it. Models remain frozen; all test outcomes are audit-only.

The time-ordered design respects information availability but does not make rows exchangeable. Shared day/hour structure, overlapping operational patterns and the explicit change limit statistical interpretation. Do not turn empirical coverage into a guarantee for a future service, and do not treat the fixed synthetic mechanism as a population sample.

## Packaged bytes and raw evidence

The [committed dataset](../reports/data/deliveries.csv.gz) is a deterministic gzip of LF CSV. [Metrics](../reports/metrics.json) record its compressed/uncompressed SHA-256, generator inputs, counts, model parameters, calibration ranks and runtime. [Predictions](../reports/predictions.csv) contain every test row's target, point prediction and interval for both estimators. Outcomes are present for offline auditing; their presence in a CSV does not make them valid serving features.

The generator and report are reproducible without network access after installing dependencies. CPU libraries/platforms can produce small floating-point differences; raw hashes identify the exact saved run. There is no private data recovery, field experiment, deployment, revenue calculation or historical employer result behind this dataset.

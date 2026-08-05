# Artifact provenance and licenses

These files are selected evaluation receipts and failure evidence, not bundled
training datasets. The repository AGPL license does not replace upstream data
terms.

| Path | Source | Terms |
|---|---|---|
| `portfolio-detection/a1.json`, `verify-final/a1/` | Pothole-600 evaluation | MIT on the author-linked Kaggle data card. Raw source frames are omitted from the current portfolio checkout. |
| `portfolio-detection/cross-domain.json`, `cross-dataset-pothole/` | Mendeley Pothole Videos v2 | CC BY 4.0. |
| `a3-grid/` and `portfolio-stereo/` | Rui Fan stereo pothole dataset | MIT; see `third_party_licenses/FAN_STEREO_MIT.txt`. |
| `vo-drift-final/`, `landmark-reid/`, `uturn-b3/`, `garage-localization*`, `gps-fusion-round5-vo/`, `system-fps-b7/`, audits derived from 4Seasons | 4Seasons | CC BY-NC-SA 4.0, non-commercial. |

See `THIRD_PARTY_NOTICES.md` for authors, source links, and additional use
restrictions. Numeric receipts may still encode derived measurements and must
retain their source attribution when redistributed.

## Benchmark ID legend

Directory and receipt names carry short benchmark IDs inherited from the
project's original evaluation plan. They are internal labels only:

| ID | Benchmark |
|---|---|
| A1 | Pothole detection accuracy (box/mask mAP) |
| A2 | Metric pothole depth and surface-area error |
| A3 | Stereo pothole pipeline end-to-end throughput |
| B1 | Stereo VO drift over 500 m windows |
| B2 | Landmark re-identification recall |
| B3 | U-turn detection latency |
| B5 | Parking-garage localization error |
| B6 | GPS handover detection |
| B7 | Localization-stack CPU throughput |
| B8 | GPS re-lock accuracy after outage |

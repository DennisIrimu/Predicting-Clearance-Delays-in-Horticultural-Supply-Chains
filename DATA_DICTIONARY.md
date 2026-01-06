# Synthetic Trade-like Dataset (Delay Prediction) — 5000 consignments (Realistic)

This dataset is **synthetic** and Trade-inspired: horticulture consignments + documents + event timelines.
It contains **no live or proprietary records**.

## Class balance
- Delay rate (`delayed_flag=1`): **48.4%**

## Files
- `consignments_5000.csv`
- `documents_5000.csv`
- `events_5000.csv`
- `tlip_like_consignments_5000.jsonl`

## Target definition
- `delayed_flag` = 1 if `total_processing_hours` > (`sla_hours` + 28)
- `delay_hours` = max(0, `total_processing_hours - (sla_hours + 28)`)


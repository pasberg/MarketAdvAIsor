#!/usr/bin/env bash
# Keep the site fresh during market hours without relying on GitHub's cron:
# publish every 5 minutes, then hand over to a new run before the 6-hour job limit.
#   every 5 min   intra (intraday bars, last price, index strip)
#   :00 and :30   + hour (hourly bars and 5-day intraday history)
#   15:50, 20:35  + daily (after the Stockholm and US closes)  [UTC]
# Market window: weekdays 06:00–20:59 UTC (08:00–22:59 Swedish summer time).
set -uo pipefail

max_seconds="${MAX_SECONDS:-20400}"   # 5 h 40 min, below the 6 h job limit
deadline=$(( $(date +%s) + max_seconds ))
in_window() { local d h; d=$(date -u +%u); h=$((10#$(date -u +%H))); [ "$d" -le 5 ] && [ "$h" -ge 6 ] && [ "$h" -le 20 ]; }
done_daily=""

while in_window && [ "$(date +%s)" -lt "$deadline" ]; do
  hm=$(date -u +%H%M); min=$((10#$(date -u +%M)))
  parts="intra"
  [ $((min % 30)) -lt 5 ] && parts="$parts,hour"
  for slot in 1550 2035; do
    if [ "$((10#$hm))" -ge "$((10#$slot))" ] && [[ "$done_daily" != *"$slot"* ]]; then
      parts="$parts,hour,daily"; done_daily="$done_daily $slot"
    fi
  done
  bash pipeline/publish.sh "$parts" || echo "::warning::Publicering misslyckades ($parts) — försöker igen om 5 min"
  # sleep until the next 5-minute mark
  now=$(date +%s); sleep $(( 300 - now % 300 + 5 ))
done

if in_window; then
  echo "Tidsgränsen nådd under börstid — startar en ny körning."
  gh workflow run live.yml --ref "$GITHUB_REF_NAME"
else
  echo "Utanför börstid — stannar."
fi

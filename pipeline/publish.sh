#!/usr/bin/env bash
# Fetch the given data parts, encrypt all parts and publish the site to gh-pages.
# Usage: pipeline/publish.sh intra,hour   (needs GH_TOKEN and GITHUB_REPOSITORY; SITE_USERS for login)
set -euo pipefail
parts="${1:-intra,hour,daily}"

# a part that was never fetched (e.g. empty cache), or lacks instruments added to
# config/universe.csv, is fetched as well
missing="$(python pipeline/fetch_market_data.py --out-dir data --missing-parts)"
[ -n "$missing" ] && parts="$parts,$missing"
parts="$(echo "$parts" | tr ',' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)"
python pipeline/fetch_market_data.py --out-dir data --parts "$parts"

# log today's recommendations and follow up earlier ones (history survives a lost cache)
(cd pipeline && python restore_log.py ../data/log.json) || echo "::warning::Kunde inte återställa loggen"
node pipeline/log_picks.js --data data --page mockup/index.html || echo "::warning::Loggningen misslyckades"
# strategy lab: re-run when daily prices were refreshed (or it never ran)
if [[ "$parts" == *daily* ]] || [ ! -f data/lab.json ]; then
  python pipeline/strategy_lab.py --data data || echo "::warning::Strategilabbet misslyckades"
fi

# intraday archive: after each daily fetch (17:50 and 22:35), keep today's 5-minute bars
if [[ "$parts" == *daily* ]]; then
  python pipeline/intraday_archive.py --data data --export data/archive_bars.json || echo "::warning::Intradagsarkivet misslyckades"
  # measured hit rates per horizon and score, shown instead of a rule of thumb
  node pipeline/calibrate.js --data data || echo "::warning::Kalibreringen misslyckades"
  # intraday strategies on the archive
  python pipeline/intraday_lab.py --data data || echo "::warning::Intradagslabbet misslyckades"
fi
# trading journals synced from the page (journal branch) -> encrypted site data
python pipeline/journal_export.py data/journal.json || echo "::warning::Dagboken kunde inte exporteras"

rm -rf site
python pipeline/encrypt_data.py --src data --dst site/data
python pipeline/build_site.py --src mockup/index.html --out site/index.html
cp mockup/analysis.js site/analysis.js

cd site
git init -q -b gh-pages
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -q -m "Publish site ($(date -u +%Y-%m-%dT%H:%MZ), $parts)"
# a single-commit branch: history does not grow with every 5-minute update
git push -q --force "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" gh-pages
echo "Publicerat: $parts"

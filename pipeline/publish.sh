#!/usr/bin/env bash
# Fetch the given data parts, encrypt all parts and publish the site to gh-pages.
# Usage: pipeline/publish.sh intra,hour   (needs GH_TOKEN and GITHUB_REPOSITORY; SITE_USERS for login)
set -euo pipefail
parts="${1:-intra,hour,daily}"

# a part that was never fetched (e.g. empty cache) is fetched as well
for p in intra hour daily; do [ -f "data/$p.json" ] || parts="$parts,$p"; done
python pipeline/fetch_market_data.py --out-dir data --parts "$parts"

rm -rf site
python pipeline/encrypt_data.py --src data --dst site/data
python pipeline/build_site.py --src mockup/index.html --out site/index.html

cd site
git init -q -b gh-pages
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -q -m "Publish site ($(date -u +%Y-%m-%dT%H:%MZ), $parts)"
# a single-commit branch: history does not grow with every 5-minute update
git push -q --force "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" gh-pages
echo "Publicerat: $parts"

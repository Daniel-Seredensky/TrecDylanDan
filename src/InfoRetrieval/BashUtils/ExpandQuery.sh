#!/usr/bin/env bash
# -------------------------
# expand_query.sh  "<query>"
# Prints one token per line (unique, lower‑cased) produced by
# the analyzer that references your synonym map.
# Requires: jq, curl, and these env vars:
#   AZURE_SEARCH_SERVICE  (e.g., "mysearchsvc")
#   AZURE_SEARCH_INDEX    (e.g., "docs")
#   AZURE_SEARCH_KEY      (admin or query key)
#   ANALYZER_NAME         (custom analyzer tied to the synonym map)
# -------------------------

set -euo pipefail

QUERY_TEXT="$1"
API_VERSION="2024-07-01"

curl -s -X POST \
  "https://${AZURE_SEARCH_SERVICE}.search.windows.net/indexes/${AZURE_SEARCH_INDEX}/analyze?api-version=${API_VERSION}" \
  -H "api-key: ${AZURE_SEARCH_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"${QUERY_TEXT}\",\"analyzer\":\"${ANALYZER_NAME}\"}" |
  jq -r '.tokens[].token' | sort -u

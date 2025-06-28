#!/usr/bin/env bash
#
# Delete **all** Assistant‑v2 threads in the current Azure OpenAI resource.
# ▸ Requires: curl, jq   (brew install jq on macOS)
# ▸ Env vars:
#       AZURE_OPENAI_ENDPOINT   # e.g. https://my‑resource.openai.azure.com
#       AZURE_OPENAI_API_KEY    # your key or a bearer token
#       API_VERSION             # defaults to latest preview that exposes /threads
#
set -euo pipefail


API_VER="2025-04-01-preview"   

page="${ENDPOINT}/openai/threads?api-version=${API_VER}"

while [[ -n "$page" && "$page" != "null" ]]; do
    echo "• Fetching thread page: $page"
    resp=$(curl -sS -H "api-key: $KEY" "$page")

    # Delete every thread in the current page
    echo "$resp" | jq -r '.data[].id' | while read -r tid; do
        echo "  → deleting $tid"
        curl -sS -X DELETE \
             -H "api-key: $KEY" \
             "${ENDPOINT}/openai/threads/${tid}?api-version=${API_VER}" \
             | jq -c '.'
    done

    # Follow pagination if the service returned a next page URL
    page=$(echo "$resp" | jq -r '.next_url // ""')
done

echo "✅  All threads deleted."

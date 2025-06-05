#!/usr/bin/env bash
# AzureJanitor.bash

set -e

# Load environment variables
source .env

if [[ -z "$AZURE_SEARCH_ENDPOINT" || -z "$SEARCH_ADMIN_KEY" || -z "$SEARCH_INDEX" ]]; then
  echo "ERROR: AZURE_SEARCH_ENDPOINT, SEARCH_ADMIN_KEY, and SEARCH_INDEX must be set."
  exit 1
fi

# Expect the schema path as the first argument
if [[ -z "$1" ]]; then
  echo "Usage: $0 <path-to-index_schema.json>"
  exit 1
fi

schema_path="$1"
if [[ ! -f "$schema_path" ]]; then
  echo "ERROR: Schema file not found at '$schema_path'"
  exit 1
fi

echo "→ Deleting index '$SEARCH_INDEX' ..."
curl -s -X DELETE \
     "$AZURE_SEARCH_ENDPOINT/indexes/$SEARCH_INDEX?api-version=2021-04-30-Preview" \
     -H "api-key: $SEARCH_ADMIN_KEY" \
     -o /dev/null -w "Status: %{http_code}\n"

echo "→ Recreating index '$SEARCH_INDEX' from schema file '$schema_path'..."
curl -s -X PUT \
     "$AZURE_SEARCH_ENDPOINT/indexes/$SEARCH_INDEX?api-version=2021-04-30-Preview" \
     -H "Content-Type: application/json" \
     -H "api-key: $SEARCH_ADMIN_KEY" \
     --data-binary "@$schema_path" \
     -o /dev/null -w "Status: %{http_code}\n"

echo "Done. '$SEARCH_INDEX' is now empty."

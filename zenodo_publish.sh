#!/usr/bin/env bash
# Zenodo deposit of this repository, from the metadata in .zenodo.json.
#
#   export ZENODO_TOKEN=<token>
#   bash zenodo_publish.sh rascunho             # create the draft, set metadata, print the reserved DOI
#   bash zenodo_publish.sh enviar <deposit_id>  # upload `git archive` of HEAD, the versioned files only
#   bash zenodo_publish.sh publicar <deposit_id>
#
# The archive comes from git, so local folders kept out of version control never reach the deposit.
set -euo pipefail

if [ -z "${ZENODO_TOKEN:-}" ]; then
  echo "Error: ZENODO_TOKEN environment variable is not set." >&2
  exit 1
fi
BASE="https://zenodo.org/api"
AUTH="Authorization: Bearer $ZENODO_TOKEN"
VERSION="1.0"
cd "$(dirname "$0")"

case "${1:-}" in
  rascunho)
    DEPOSIT=$(curl -s --max-time 60 -X POST "$BASE/deposit/depositions" -H "$AUTH" -H "Content-Type: application/json" \
      -d '{"metadata": {"prereserve_doi": true}}')
    ID=$(echo "$DEPOSIT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    META=$(VERSION="$VERSION" python3 -c "
import datetime, json, os
zj = json.load(open('.zenodo.json'))
m = {k: zj[k] for k in ('title', 'description', 'upload_type', 'creators', 'keywords', 'related_identifiers')}
m.update(version=os.environ['VERSION'], license=zj['license'].lower(), access_right='open',
         publication_date=datetime.date.today().isoformat(), prereserve_doi=True)
print(json.dumps({'metadata': m}))")
    curl -s --max-time 60 -X PUT "$BASE/deposit/depositions/$ID" -H "$AUTH" -H "Content-Type: application/json" -d "$META" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get('metadata',{}); print('deposit:', d.get('id')); print('title:', m.get('title')); print('reserved DOI:', m.get('prereserve_doi',{}).get('doi')); print('errors:', d.get('errors'))"
    ;;
  enviar)
    ID="$2"
    ZIP="$(mktemp -d)/homebrewed-slms-$VERSION.zip"
    git archive --format=zip --prefix="homebrewed-slms-$VERSION/" -o "$ZIP" HEAD
    BUCKET=$(curl -s --max-time 60 "$BASE/deposit/depositions/$ID" -H "$AUTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['links']['bucket'])")
    curl -s --max-time 1800 -X PUT "$BUCKET/$(basename "$ZIP")" -H "$AUTH" -H "Content-Type: application/octet-stream" --data-binary "@$ZIP" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print('uploaded:', d.get('key'), d.get('size'), 'bytes', d.get('checksum'))"
    sha256sum "$ZIP"
    rm -f "$ZIP"
    ;;
  publicar)
    ID="$2"
    curl -s --max-time 120 -X POST "$BASE/deposit/depositions/$ID/actions/publish" -H "$AUTH" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print('DOI:', d.get('doi')); print('concept DOI:', d.get('conceptdoi')); print('record:', d.get('links',{}).get('record_html')); print('errors:', d.get('errors'))"
    ;;
  *)
    echo "usage: bash zenodo_publish.sh rascunho | enviar <deposit_id> | publicar <deposit_id>" >&2
    exit 1
    ;;
esac

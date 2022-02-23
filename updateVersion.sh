#! /bin/bash

curSubVersion=$(poetry version -s | grep -Eo '[0-9]+$')
mainVersion=$(poetry version -s | grep -Eo '[0-9]+\.[0-9]+\.')
newSubVersion=$((curSubVersion+1))
newVersion="$mainVersion$newSubVersion"
poetry version ${newVersion}


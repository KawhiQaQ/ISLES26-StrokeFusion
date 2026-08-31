#!/bin/bash
set -e
exec /tmp/isles26_pybootstrap/ld-linux-x86-64.so.2 \
  --library-path /tmp/isles26_pybootstrap/usr/lib/x86_64-linux-gnu \
  --argv0 /tmp/isles26_pybootstrap/usr/bin/python3.12 \
  /tmp/isles26_pybootstrap/python3.12.real "$@"

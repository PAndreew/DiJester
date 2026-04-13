#!/bin/sh
# Fix data volume ownership (mounted as root), then drop to UID 1000
chown -R 1000:1000 /app/data /home/appuser/.cache 2>/dev/null || true
exec gosu 1000 python3 /app/app.py

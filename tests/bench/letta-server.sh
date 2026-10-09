#!/usr/bin/env bash
# TEST HARNESS ONLY: starts the Letta server that tests/bench/memory_letta.py talks to (letta 0.16.8, Apache-2.0,
# https://github.com/letta-ai/letta), the way Letta's own image does it (letta/server/startup.sh and init.sql at tag
# 0.16.8): a PostgreSQL with pgvector, the release's `alembic upgrade head`, then `letta server`. Letta 0.16.8 needs
# Postgres (its ORM imports asyncpg; SQLite is no longer wired in), so here it is a local cluster in $DATA run as
# the calling user, instead of the one bundled in Letta's image. Redis is optional in Letta and not started.
#
#   bash tests/bench/letta-server.sh   # http://127.0.0.1:8283; Postgres on 127.0.0.1:5432; data in /work/letta
#
# Once, as root:  apt-get install postgresql-16 postgresql-16-pgvector && python3 -m venv /opt/mem-letta &&
#                 /opt/mem-letta/bin/pip install -r tests/bench/requirements-letta.txt
# No model key is needed here: memory_letta.py points every agent at the metering proxy, which adds the key.
set -euo pipefail

VENV=/opt/mem-letta
DATA=${LETTA_BENCH_DIR:-/work/letta}
PORT=${LETTA_PORT:-8283}
PG_PORT=${LETTA_PG_PORT:-5432}
PG=/usr/lib/postgresql/16/bin
SRC=$VENV/src/letta-0.16.8

# The migrations ship in the release's sdist, not in the wheel.
if [ ! -f "$SRC/alembic.ini" ]; then
    mkdir -p "$VENV/src"
    "$VENV/bin/pip" download -q --no-deps --no-binary :all: letta==0.16.8 -d "$VENV/src"
    tar xzf "$VENV/src/letta-0.16.8.tar.gz" -C "$VENV/src" letta-0.16.8/alembic.ini letta-0.16.8/alembic
fi

mkdir -p "$DATA"
if [ ! -s "$DATA/pg/PG_VERSION" ]; then
    "$PG/initdb" -D "$DATA/pg" -U letta --auth=trust >/dev/null
fi
if ! "$PG/pg_ctl" -D "$DATA/pg" status >/dev/null 2>&1; then
    "$PG/pg_ctl" -D "$DATA/pg" -l "$DATA/postgres.log" -w \
        -o "-p $PG_PORT -k $DATA -c listen_addresses=127.0.0.1 -c max_connections=200" start >/dev/null
fi
psql() { "$PG/psql" -h 127.0.0.1 -p "$PG_PORT" -U letta -v ON_ERROR_STOP=1 -qtA "$@"; }
if [ -z "$(psql -d postgres -c "SELECT 1 FROM pg_database WHERE datname = 'letta'")" ]; then
    psql -d postgres -c "CREATE DATABASE letta"
    # init.sql
    psql -d letta -c "CREATE SCHEMA letta AUTHORIZATION letta" \
        -c "ALTER DATABASE letta SET search_path TO letta" \
        -c "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA letta" \
        -c "DROP SCHEMA IF EXISTS public CASCADE"
fi

export LETTA_PG_URI="postgresql://letta:letta@127.0.0.1:$PG_PORT/letta"
export LETTA_DIR="$DATA"
(cd "$SRC" && "$VENV/bin/alembic" upgrade head)
exec "$VENV/bin/letta" server --host 127.0.0.1 --port "$PORT"

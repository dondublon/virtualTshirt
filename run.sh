#!/usr/bin/env bash
# Запуск можно делать из любого каталога: скрипт сам переходит в свою папку,
# чтобы uv нашёл pyproject.toml/uv.lock (и, значит, .venv проекта) и main.py.
# Каталог скрипта определяем вручную (POSIX sh, без BASH_SOURCE) и разыменовываем
# симлинки — чтобы работал и вызов вида `sh run.sh`, и запуск через ссылку из PATH.
set -eu

SCRIPT_PATH="$0"
while [ -L "$SCRIPT_PATH" ]; do
    link="$(readlink "$SCRIPT_PATH")"
    case "$link" in
        /*) SCRIPT_PATH="$link" ;;
        *) SCRIPT_PATH="$(dirname -- "$SCRIPT_PATH")/$link" ;;
    esac
done
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run main.py --camera 2 --output /dev/video10

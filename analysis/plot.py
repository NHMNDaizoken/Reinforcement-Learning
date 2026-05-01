"""Generate Phase 3 training and ATL analysis plots."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import struct
import zlib
from pathlib import Path
from typing import Iterable

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError:
    plt = None
    pd = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "analysis"
CHUNKSIZE = 100_000

EPISODE_COLUMNS = ("episode", "episode_id", "run_episode")
REWARD_COLUMNS = ("reward", "total_reward", "episode_reward")
ATL_COLUMNS = ("atl", "average_travel_time", "avg_travel_time", "travel_time")
LOSS_COLUMNS = ("loss", "td_loss", "huber_loss", "q_loss")
SOURCE_COLUMNS = ("policy", "source", "algorithm")


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_basic_png(path: Path, series: list[list[float]], width: int = 640, height: int = 360) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = bytearray([255, 255, 255] * width * height)

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            pixels[idx : idx + 3] = bytes(color)

    for x in range(50, width - 30):
        set_pixel(x, height - 45, (190, 190, 190))
    for y in range(25, height - 44):
        set_pixel(50, y, (190, 190, 190))

    values = [value for line in series for value in line if value == value]
    if values:
        min_value = min(values)
        max_value = max(values)
        span = max(max_value - min_value, 1.0)
        colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44)]
        for line_index, line in enumerate(series):
            if not line:
                continue
            points = []
            for idx, value in enumerate(line):
                x = 50 + round(idx * (width - 90) / max(len(line) - 1, 1))
                y = height - 45 - round((value - min_value) * (height - 80) / span)
                points.append((x, y))
            color = colors[line_index % len(colors)]
            for x, y in points:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        set_pixel(x + dx, y + dy, color)

    raw = bytearray()
    for y in range(height):
        start = y * width * 3
        raw.append(0)
        raw.extend(pixels[start : start + width * 3])

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _summarize_records(records: Iterable[dict[str, str]], source: str) -> list[dict[str, float | str | bool]]:
    aggregates: dict[int, dict[str, float | int | bool]] = {}
    for row in records:
        columns = list(row)
        episode_col = _first_present(columns, EPISODE_COLUMNS)
        reward_col = _first_present(columns, REWARD_COLUMNS)
        atl_col = _first_present(columns, ATL_COLUMNS)
        loss_col = _first_present(columns, LOSS_COLUMNS)
        episode = int(float(row.get(episode_col or "", 0) or 0))
        item = aggregates.setdefault(
            episode,
            {"reward": 0.0, "atl_sum": 0.0, "atl_count": 0, "loss_sum": 0.0, "loss_count": 0, "atl_proxy": False},
        )
        if reward_col:
            reward = float(row.get(reward_col, 0.0) or 0.0)
            item["reward"] = float(item["reward"]) + reward
        else:
            reward = 0.0
        if atl_col:
            item["atl_sum"] = float(item["atl_sum"]) + float(row.get(atl_col, 0.0) or 0.0)
            item["atl_count"] = int(item["atl_count"]) + 1
        elif reward_col:
            item["atl_sum"] = float(item["atl_sum"]) + -reward
            item["atl_count"] = int(item["atl_count"]) + 1
            item["atl_proxy"] = True
        if loss_col and row.get(loss_col, "") != "":
            item["loss_sum"] = float(item["loss_sum"]) + float(row[loss_col])
            item["loss_count"] = int(item["loss_count"]) + 1

    rows = []
    for episode, item in sorted(aggregates.items()):
        atl_count = int(item["atl_count"])
        loss_count = int(item["loss_count"])
        rows.append(
            {
                "source": source,
                "episode": episode,
                "reward": float(item["reward"]),
                "atl": float(item["atl_sum"]) / atl_count if atl_count else float("nan"),
                "loss": float(item["loss_sum"]) / loss_count if loss_count else float("nan"),
                "atl_proxy": bool(item["atl_proxy"]),
            }
        )
    return rows


def _load_csv_basic(path: Path, source: str) -> list[dict[str, float | str | bool]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return _summarize_records(csv.DictReader(file), source)


def _load_sqlite_basic(path: Path, source: str) -> list[dict[str, float | str | bool]]:
    if not path.exists():
        return []
    rows: list[dict[str, float | str | bool]] = []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        for table in _sqlite_tables(connection):
            columns = _sqlite_columns(connection, table)
            if not _table_matches_source(table, columns, source):
                continue
            source_col = _first_present(columns, SOURCE_COLUMNS)
            where = f' WHERE lower("{source_col}") = ?' if source_col else ""
            params = (source.lower(),) if source_col else ()
            query = f'SELECT * FROM "{table}"{where}'
            records = (dict(row) for row in connection.execute(query, params))
            rows.extend(_summarize_records(records, source))
    return rows


def _write_plots_basic(data_dir: Path, output_dir: Path) -> list[Path]:
    dqn = _load_csv_basic(data_dir / "buffer_dqn.csv", "dqn") or _load_sqlite_basic(
        data_dir / "offline_dataset.db", "dqn"
    )
    baseline = _load_csv_basic(data_dir / "buffer_baseline.csv", "baseline") or _load_sqlite_basic(
        data_dir / "offline_dataset.db", "baseline"
    )
    outputs = [output_dir / "learning_curve.png", output_dir / "atl_comparison.png"]
    _write_basic_png(outputs[0], [[float(row["reward"]) for row in dqn]])
    _write_basic_png(
        outputs[1],
        [[float(row["atl"]) for row in dqn], [float(row["atl"]) for row in baseline]],
    )
    return outputs


def _empty_series(source: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=["source", "episode", "reward", "atl", "loss", "atl_proxy"]
    ).astype({"source": "object"})


def _summarize_rows(rows: pd.DataFrame, source: str) -> pd.DataFrame:
    if rows.empty:
        return _empty_series(source)

    episode_col = _first_present(rows.columns, EPISODE_COLUMNS)
    reward_col = _first_present(rows.columns, REWARD_COLUMNS)
    atl_col = _first_present(rows.columns, ATL_COLUMNS)
    loss_col = _first_present(rows.columns, LOSS_COLUMNS)

    if episode_col is None:
        rows = rows.copy()
        rows["_episode"] = 0
        episode_col = "_episode"

    summary = pd.DataFrame()
    summary["episode"] = pd.to_numeric(rows[episode_col], errors="coerce").fillna(0).astype(int)

    if reward_col is not None:
        summary["reward"] = pd.to_numeric(rows[reward_col], errors="coerce")
    elif atl_col is not None:
        summary["reward"] = -pd.to_numeric(rows[atl_col], errors="coerce")
    else:
        summary["reward"] = pd.NA

    if atl_col is not None:
        summary["atl"] = pd.to_numeric(rows[atl_col], errors="coerce")
        summary["atl_proxy"] = False
    elif reward_col is not None:
        summary["atl"] = -pd.to_numeric(rows[reward_col], errors="coerce")
        summary["atl_proxy"] = True
    else:
        summary["atl"] = pd.NA
        summary["atl_proxy"] = True

    if loss_col is not None:
        summary["loss"] = pd.to_numeric(rows[loss_col], errors="coerce")
    else:
        summary["loss"] = pd.NA

    grouped = summary.groupby("episode", as_index=False).agg(
        reward=("reward", "sum"),
        atl=("atl", "mean"),
        loss=("loss", "mean"),
        atl_proxy=("atl_proxy", "max"),
    )
    grouped.insert(0, "source", source)
    return grouped.sort_values("episode").reset_index(drop=True)


def _combine_summaries(summaries: list[pd.DataFrame], source: str) -> pd.DataFrame:
    frames = [frame for frame in summaries if not frame.empty]
    if not frames:
        return _empty_series(source)

    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("episode", as_index=False).agg(
        reward=("reward", "sum"),
        atl=("atl", "mean"),
        loss=("loss", "mean"),
        atl_proxy=("atl_proxy", "max"),
    )
    grouped.insert(0, "source", source)
    return grouped.sort_values("episode").reset_index(drop=True)


def load_policy_csv(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return _empty_series(source)

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = {
        column
        for column in (
            _first_present(columns, EPISODE_COLUMNS),
            _first_present(columns, REWARD_COLUMNS),
            _first_present(columns, ATL_COLUMNS),
            _first_present(columns, LOSS_COLUMNS),
        )
        if column is not None
    }
    if not wanted:
        return _empty_series(source)

    chunks = pd.read_csv(path, usecols=sorted(wanted), chunksize=CHUNKSIZE)
    return _combine_summaries([_summarize_rows(chunk, source) for chunk in chunks], source)


def _sqlite_tables(connection: sqlite3.Connection) -> list[str]:
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    return [row[0] for row in connection.execute(query).fetchall()]


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _table_matches_source(table: str, columns: list[str], source: str) -> bool:
    source_col = _first_present(columns, SOURCE_COLUMNS)
    if source_col is not None:
        return True
    return source.lower() in table.lower()


def load_policy_sqlite(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return _empty_series(source)

    summaries: list[pd.DataFrame] = []
    with sqlite3.connect(path) as connection:
        for table in _sqlite_tables(connection):
            columns = _sqlite_columns(connection, table)
            if not _table_matches_source(table, columns, source):
                continue

            selected = [
                column
                for column in (
                    _first_present(columns, EPISODE_COLUMNS),
                    _first_present(columns, REWARD_COLUMNS),
                    _first_present(columns, ATL_COLUMNS),
                    _first_present(columns, LOSS_COLUMNS),
                    _first_present(columns, SOURCE_COLUMNS),
                )
                if column is not None
            ]
            if not selected:
                continue

            source_col = _first_present(columns, SOURCE_COLUMNS)
            where = ""
            if source_col is not None:
                where = f' WHERE lower("{source_col}") = ?'
                params: tuple[str, ...] = (source.lower(),)
            else:
                params = ()

            select_list = ", ".join(f'"{column}"' for column in selected)
            query = f'SELECT {select_list} FROM "{table}"{where}'
            for chunk in pd.read_sql_query(query, connection, params=params, chunksize=CHUNKSIZE):
                summaries.append(_summarize_rows(chunk, source))

    return _combine_summaries(summaries, source)


def load_phase3_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    dqn = load_policy_csv(data_dir / "buffer_dqn.csv", "dqn")
    baseline = load_policy_csv(data_dir / "buffer_baseline.csv", "baseline")
    sqlite_path = data_dir / "offline_dataset.db"

    if dqn.empty:
        dqn = load_policy_sqlite(sqlite_path, "dqn")
    if baseline.empty:
        baseline = load_policy_sqlite(sqlite_path, "baseline")

    frames = [frame for frame in (dqn, baseline) if not frame.empty]
    if not frames:
        return _empty_series("none")
    return pd.concat(frames, ignore_index=True)


def _plot_learning_curve(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5), dpi=120)

    dqn = data[data["source"] == "dqn"].sort_values("episode")
    if dqn.empty:
        axis.text(0.5, 0.5, "No DQN data available", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.plot(dqn["episode"], dqn["reward"], marker="o", linewidth=1.8, label="DQN reward")
        axis.set_xlabel("Episode")
        axis.set_ylabel("Episode reward")
        axis.grid(True, alpha=0.3)

        loss = dqn.dropna(subset=["loss"])
        if not loss.empty:
            loss_axis = axis.twinx()
            loss_axis.plot(
                loss["episode"],
                loss["loss"],
                color="#d62728",
                linestyle="--",
                marker="s",
                linewidth=1.4,
                label="DQN loss",
            )
            loss_axis.set_ylabel("Loss")
            handles, labels = axis.get_legend_handles_labels()
            loss_handles, loss_labels = loss_axis.get_legend_handles_labels()
            axis.legend(handles + loss_handles, labels + loss_labels, loc="best")
        else:
            axis.legend(loc="best")

    axis.set_title("DQN Learning Curve")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_atl_comparison(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5), dpi=120)

    atl_data = data.dropna(subset=["atl"])
    if atl_data.empty:
        axis.text(0.5, 0.5, "No ATL or reward data available", ha="center", va="center")
        axis.set_axis_off()
    else:
        ordered_sources = [source for source in ("dqn", "baseline") if source in set(atl_data["source"])]
        values = [atl_data.loc[atl_data["source"] == source, "atl"].tolist() for source in ordered_sources]
        axis.boxplot(
            values,
            tick_labels=[source.upper() for source in ordered_sources],
            patch_artist=True,
        )
        axis.set_ylabel("Average travel time" + (" (reward proxy)" if atl_data["atl_proxy"].any() else ""))
        axis.grid(True, axis="y", alpha=0.3)

    axis.set_title("ATL Comparison")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_plots(data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    if pd is None or plt is None:
        return _write_plots_basic(data_dir, output_dir)

    data = load_phase3_data(data_dir)
    outputs = [
        output_dir / "learning_curve.png",
        output_dir / "atl_comparison.png",
    ]
    _plot_learning_curve(data, outputs[0])
    _plot_atl_comparison(data, outputs[1])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 analysis plots.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    for output in write_plots(args.data_dir, args.output_dir):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()

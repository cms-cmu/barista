#!/usr/bin/env python
"""Commits / MRs merged per day on origin/master for barista and coffea4bees,
split human vs Claude-involved (Co-Authored-By: Claude trailer).

Rerunnable: fetch -> count -> plot. Run on the host (needs git + matplotlib + pandas):
    python src/tools/repo_activity.py [--since 2025-09-01] [--outdir DIR] [--no-fetch]

Read-only w.r.t. the repos (only `git fetch origin`). See src/tools/README.md.
"""
import argparse
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

BARISTA = Path(__file__).resolve().parents[2]
REPO_PATHS = {"coffea4bees": BARISTA / "coffea4bees", "barista": BARISTA}  # overridable via CLI
REPOS = ["coffea4bees", "barista"]          # coffea4bees first: barista forked from it
OUT = Path("repo_activity_output")
REF = "origin/master"
TRAILER = re.compile(r"^co-authored-by:\s*claude", re.I | re.M)

C_HUMAN = "#5B8DD6"
C_CLAUDE = "#D9734E"
INK, INK2, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(REPO_PATHS[repo]), *args],
                          check=True, capture_output=True, text=True).stdout


def log(repo, *extra):
    """[(sha, author_date, is_claude, parents)] via NUL/RS-delimited log."""
    raw = git(repo, "log", REF, *extra, "--date=short",
              "--format=%H%x00%ad%x00%P%x00%B%x1e")
    rows = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, date, parents, body = rec.split("\x00", 3)
        rows.append((sha, date, bool(TRAILER.search(body)), parents.split()))
    return rows


def commit_table(seen):
    """Non-merge commits per repo; a commit already counted in an earlier repo
    (shared pre-fork history) is skipped."""
    out = []
    for repo in REPOS:
        for sha, date, claude, _ in log(repo, "--no-merges"):
            if sha in seen:
                continue
            seen.add(sha)
            out.append((repo, date, claude))
    return pd.DataFrame(out, columns=["repo", "date", "claude"])


def mr_table(seen):
    """First-parent merge commits on master = MRs merged. Claude-involved if the
    merge message or any commit in merge^1..merge^2 carries the trailer."""
    out = []
    for repo in REPOS:
        claude_shas = {s for s, _, c, _ in log(repo) if c}
        for sha, date, claude, parents in log(repo, "--first-parent", "--merges"):
            if sha in seen:
                continue
            seen.add(sha)
            branch = git(repo, "rev-list", f"{parents[0]}..{parents[1]}").split()
            out.append((repo, date, claude or any(b in claude_shas for b in branch)))
    return pd.DataFrame(out, columns=["repo", "date", "claude"])


def daily(df, repo, since, until):
    d = df[df.repo == repo].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d.date >= since) & (d.date <= until)]
    idx = pd.date_range(since, until, freq="D")
    t = (d.groupby(["date", "claude"]).size().unstack(fill_value=0)
          .reindex(columns=[False, True], fill_value=0).reindex(idx, fill_value=0))
    t.columns = ["human", "claude"]
    t["total"] = t.human + t.claude
    t["avg7"] = t.total.rolling(7, min_periods=1).mean()
    return t


def setup_fonts():
    """Use Inter if installed (user or system font dirs), else matplotlib's default."""
    for d in (Path.home() / "Library/Fonts", Path.home() / ".local/share/fonts",
              Path("/usr/share/fonts")):
        for f in d.rglob("Inter-*.[ot]tf") if d.is_dir() else ():
            font_manager.fontManager.addfont(str(f))
    family = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "sans-serif"
    plt.rcParams.update({"font.family": family, "font.size": 11,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})


def plot(tables, title, subtitle, unit, fname):
    fig, axes = plt.subplots(len(REPOS), 1, figsize=(14, 7), sharex=True,
                             facecolor="white", gridspec_kw={"hspace": 0.18})
    fig.subplots_adjust(left=0.06, right=0.87, top=0.86, bottom=0.07)
    fig.suptitle(title, x=0.465, y=0.975, fontsize=22, fontweight="bold", color=INK)
    fig.text(0.465, 0.915, subtitle, ha="center", fontsize=12.5, color=INK2)
    for ax, repo in zip(axes, REPOS):
        t = tables[repo]
        x = t.index
        ax.bar(x, t.human, width=0.8, color=C_HUMAN, linewidth=0)
        ax.bar(x, t.claude, width=0.8, bottom=t.human, color=C_CLAUDE, linewidth=0)
        ax.plot(x, t.avg7, color=INK, lw=1.8)
        ax.set_facecolor("white")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.005, 0.97, repo, transform=ax.transAxes, va="top",
                fontsize=14, fontweight="semibold", color=INK)
        ymax = t.total.max()
        ax.set_ylim(0, ymax * 1.28)
        ax.margins(x=0.005)

        # single callout on the peak day
        pk = t.total.idxmax()
        ax.annotate(f"$\\bf{{{ymax}}}$ {unit}\non {pk:%b %-d, %Y}",
                    xy=(pk, ymax), xytext=(-14, -6), textcoords="offset points",
                    ha="right", va="top", fontsize=10.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=INK2, lw=1.2))

        # end-labels instead of a legend: totals over the window + current 7-day avg
        nh, nc = int(t.human.sum()), int(t.claude.sum())
        last = t.avg7.iloc[-1]
        labels = [(f"Human  {nh:,}", C_HUMAN, 0.62),
                  (f"Claude  {nc:,}", C_CLAUDE, 0.47),
                  (f"7-day avg  {last:.1f}/day", INK, 0.32)]
        for txt, col, yf in labels:
            ax.text(1.012, yf, "■ " if col != INK else "— ", transform=ax.transAxes,
                    color=col, fontsize=11, va="center", ha="left")
            ax.text(1.032, yf, txt, transform=ax.transAxes, color=INK,
                    fontsize=11, va="center", ha="left")
        ax.plot([x[-1]], [last], "o", color=INK, ms=4, clip_on=False)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: (lambda d: f"{d:%b}\n{d:%Y}" if d.month == 1 or v == axes[-1].get_xticks()[0]
                      else f"{d:%b}")(mdates.num2date(v))))
    fig.supylabel(f"{unit[0].upper() + unit[1:]} per day", x=0.012, color=INK2, fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def plot_summed(daily_total, freq, title, subtitle, unit, fname):
    """Single series (barista + coffea4bees, no human/Claude split), per day or per week."""
    until = daily_total.index[-1]
    if freq == "W":
        s = daily_total.resample("W-MON", label="left", closed="left").sum()  # Mon-start weeks
        partial = s.index[-1] + pd.Timedelta(days=6) > until
        complete = s.iloc[:-1] if partial else s
        avg, avg_lbl, per = complete.rolling(4, min_periods=1).mean(), "4-wk avg", "week"
        x, width = s.index + pd.Timedelta(days=3), 5.5
        xavg = complete.index + pd.Timedelta(days=3)
    else:
        s = daily_total
        partial, complete = False, s
        avg, avg_lbl, per = s.rolling(7, min_periods=1).mean(), "7-day avg", "day"
        x, xavg, width = s.index, s.index, 0.8

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.855, top=0.84, bottom=0.09)
    fig.suptitle(title, x=0.47, y=0.965, fontsize=22, fontweight="bold", color=INK)
    fig.text(0.47, 0.895, subtitle, ha="center", fontsize=12.5, color=INK2)

    bars = ax.bar(x, s.values, width=width, color=C_HUMAN, linewidth=0)
    if partial:
        bars[-1].set_alpha(0.35)
        ax.annotate("this week\nso far", xy=(x[-1], s.iloc[-1]), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=INK2)
    ax.plot(xavg, avg.values, color=INK, lw=1.8)
    ax.plot([xavg[-1]], [avg.iloc[-1]], "o", color=INK, ms=4, clip_on=False)

    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ymax = complete.max()
    ax.set_ylim(0, max(ymax, s.max()) * 1.22)
    ax.margins(x=0.005)

    pk = complete.idxmax()
    when = f"week of {pk:%b %-d, %Y}" if freq == "W" else f"on {pk:%b %-d, %Y}"
    ax.annotate(f"$\\bf{{{int(ymax)}}}$ {unit}\n{when}", xy=(pk, ymax), xytext=(-14, -6),
                textcoords="offset points", ha="right", va="top", fontsize=10.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=INK2, lw=1.2))

    unit_per = "wk" if per == "week" else "day"
    for txt, mark, col, yf in [(f"Total  {int(s.sum()):,}", "■ ", C_HUMAN, 0.56),
                               (f"{avg_lbl}  {avg.iloc[-1]:.{0 if per == 'week' else 1}f}/{unit_per}",
                                "— ", INK, 0.48)]:
        ax.text(1.012, yf, mark, transform=ax.transAxes, color=col, fontsize=11, va="center")
        ax.text(1.032, yf, txt, transform=ax.transAxes, color=INK, fontsize=11, va="center")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: (lambda d: f"{d:%b}\n{d:%Y}" if d.month == 1 or v == ax.get_xticks()[0]
                      else f"{d:%b}")(mdates.num2date(v))))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel(f"{unit[0].upper() + unit[1:]} per {per}", color=INK2, fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    return dict(total=int(s.sum()), peak=int(ymax), peak_at=f"{pk:%Y-%m-%d}",
                avg_now=round(avg.iloc[-1], 2), partial_last=bool(partial))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-09-01")
    ap.add_argument("--until", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--outdir", default=str(OUT), help="where figures/CSVs go")
    ap.add_argument("--barista", default=str(REPO_PATHS["barista"]), help="barista clone")
    ap.add_argument("--coffea4bees", default=str(REPO_PATHS["coffea4bees"]),
                    help="coffea4bees clone (default: the one inside the barista workspace)")
    a = ap.parse_args()
    OUT = Path(a.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    REPO_PATHS.update(barista=Path(a.barista), coffea4bees=Path(a.coffea4bees))
    if not a.no_fetch:
        for r in REPOS:
            git(r, "fetch", "--quiet", "origin")
    setup_fonts()
    since, until = pd.Timestamp(a.since), pd.Timestamp(a.until)
    span = f"{since:%b %Y}–{until:%b %Y}"

    commits = commit_table(set())
    mrs = mr_table(set())
    summary, summed = {}, {}
    for name, df, unit, title, sub in [
        ("commits", commits, "commits", "Commits landed per day",
         f"Non-merge commits on origin/master by author date, {span} · "
         "Claude = Co-Authored-By: Claude trailer"),
        ("mrs", mrs, "MRs", "Merge requests merged per day",
         f"First-parent merge commits on origin/master, {span} · "
         "Claude = any commit in the MR carries the Claude trailer"),
    ]:
        tables = {r: daily(df, r, since, until) for r in REPOS}
        plot(tables, title, sub, unit, f"repo_activity_{name}")
        for r, t in tables.items():
            pk = t.total.idxmax()
            summary[(name, r)] = dict(human=int(t.human.sum()), claude=int(t.claude.sum()),
                                      peak=int(t.total.max()), peak_day=f"{pk:%Y-%m-%d}",
                                      peak_claude=int(t.claude[pk]),
                                      avg7_now=round(t.avg7.iloc[-1], 2),
                                      avg7_claude_now=round(t.claude.rolling(7, min_periods=1).mean().iloc[-1], 2))
        # summed over repos, no human/Claude split, per day and per week
        tot = sum(t.total for t in tables.values())
        what = ("Non-merge commits" if name == "commits" else "MR merge commits")
        for freq, per in (("D", "day"), ("W", "week")):
            sub_s = (f"{what} on origin/master, barista + coffea4bees combined "
                     f"(shared history once), {span}")
            if freq == "W":
                sub_s += " · Mon-start weeks"
            noun = "Commits landed" if name == "commits" else "Merge requests merged"
            r = plot_summed(tot, freq, f"{noun} per {per}", sub_s, unit,
                            f"summed_{name}_{'daily' if freq == 'D' else 'weekly'}")
            summed[(name, per)] = r
    print(pd.DataFrame(summed).T.to_string())
    pd.DataFrame(summed).T.to_csv(OUT / "summary_summed.csv")
    s = pd.DataFrame(summary).T
    print(s.to_string())
    s.to_csv(OUT / "summary.csv")
    print(f"wrote figures and CSVs to {OUT.resolve()}")


if __name__ == "__main__":
    main()

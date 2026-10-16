"""README media from one recorded match, offline: stills, an MP4 clip and an animated GIF (0 provider calls).

    python -m tools.make_readme_media                    # examples/replays/07_llm_saboteurs_sonnet_vs_opus.json -> docs/media/
    python -m tools.make_readme_media REPLAY --out DIR [--no-video]

The replay is re-issued frame by frame into a fresh engine with no agent (spy_vs_ai.runtime.playback), the chosen
frames are drawn offscreen by the game's own renderer (SDL dummy driver), and ffmpeg encodes the clips. Nothing is
written unless the replay reproduces its recording. Presentation timers (cues, banners) run on the replay's game
clock rather than the wall clock, so a cue lasts as long on video as it did on screen.

The clip windows and still frames below are chosen for the default replay (Claude Sonnet 5.5 vs Claude Opus 5.5 on
Bedrock, both saboteurs, level_01, seed 42, playable mode with the enhanced gameplay on).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time as _time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = ROOT / "examples/replays/07_llm_saboteurs_sonnet_vs_opus.json"
DEFAULT_OUT = ROOT / "docs/media"
LABEL = "Claude Sonnet 5.5 (Black) vs Claude Opus 5.5 (White), both saboteurs"

FPS = 60                  # engine frames per second
RENDER_EVERY = 2          # one rendered frame every 2 engine frames: 30 fps video
# (name, first frame, last frame): an item found, Black's door trap catching White, the safe and the escape
CLIPS = (("item_found", 560, 900), ("door_trap", 7560, 7900), ("escape", 12040, 12525))
GIF_CLIPS = ("door_trap", "escape")
GIF_WIDTH, GIF_FPS, GIF_COLORS = 720, 12, 128  # keeps the README GIF a few MB
HOLD_FINAL_SECONDS = 2.5  # the final ranking card at the end of the MP4
# frame -> (file, view): play = split panes, overview = the whole dressed map, research = inspector + history
STILLS = {3700: ("overview.png", "overview"), 7762: ("hero_door_trap.png", "play"),
          12130: ("research_view.png", "research")}


class GameClock:
    """time.monotonic() for the presentation layers, advanced by the replay instead of the wall clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __getattr__(self, name):
        return getattr(_time, name)

    def monotonic(self) -> float:
        return self.now


def set_view(r, view: str) -> None:
    r.overview = view == "overview"
    r.split = view != "overview"
    research = view == "research"
    r.runtime_view = "research" if research else "play"
    r.show_inspector = r.show_history = research
    if hasattr(r, "show_detail"):
        r.show_detail = research


def setup_still(path: Path) -> None:
    """The setup screen, preset to the match that the clips come from (no provider status probed)."""
    import pygame
    from spy_vs_ai.runtime import config as rc
    from spy_vs_ai.ui.setup import HEIGHT, WIDTH, SetupScreen
    cfg = rc.from_dict({"black": {"policy": "real_llm", "provider": "bedrock", "model": "us.anthropic.claude-sonnet-5-5",
                                  "personality": "saboteur"},
                        "white": {"policy": "real_llm", "provider": "bedrock", "model": "us.anthropic.claude-opus-5-5",
                                  "personality": "saboteur"},
                        "level": "level_01", "seed": 42})
    surface = pygame.Surface((WIDTH, HEIGHT))
    SetupScreen(cfg, screen=surface, provider_status={}).draw()
    pygame.image.save(surface, str(path))


def render_replay(replay: Path, frames_dir: Path, out: Path) -> dict:
    import pygame
    import workshop.live.overlay as overlay
    import spy_vs_ai.ui.enhanced as enhanced
    from spy_vs_ai.runtime import playback as pb

    clock = GameClock()
    overlay.time = enhanced.time = clock  # presentation timers only: the engine never reads them
    p = pb.Playback(pb.load(replay), label=LABEL)
    game, r = p.game, p.game.renderer
    r.speed, r.paused = 1.0, False
    clip_of = {}
    for name, first, last in CLIPS:
        for i in range(first, min(last, len(p.frames) - 1) + 1):
            clip_of[i] = name
    counts = {name: 0 for name, _, _ in CLIPS}
    stills = {}
    pending = 0.0
    while not p.done:
        frozen = p.next_frame_frozen()
        i = p.i
        p.step()
        if not frozen:
            clock.now += 1.0 / FPS
            pending += 1.0 / FPS
        clip = clip_of.get(i)
        draw_clip = clip is not None and not frozen and (i - next(f for n, f, _ in CLIPS if n == clip)) % RENDER_EVERY == 0
        still = STILLS.get(i)
        if not (draw_clip or still):
            r._collect_cues(game.state)  # keep cues and effects current without drawing
            r._update_effects(game.state, pending)
            pending = 0.0
            continue
        r.match_over = p.done
        if draw_clip:
            set_view(r, "play")
            r.render(game.state, pending, None, None, game.bot_controller)
            pending = 0.0
            pygame.image.save(game.screen, str(frames_dir / f"{clip}_{counts[clip]:05d}.png"))
            counts[clip] += 1
        if still:
            name, view = still
            set_view(r, view)
            r.render(game.state, pending, None, None, game.bot_controller)
            pending = 0.0
            stills[name] = frames_dir / name
            pygame.image.save(game.screen, str(stills[name]))
    result = p.finish()
    set_view(r, "play")
    r.match_over = True
    r.render(game.state, 0.0, None, None, game.bot_controller)
    stills["final_ranking.png"] = frames_dir / "final_ranking.png"
    pygame.image.save(game.screen, str(stills["final_ranking.png"]))
    result["clip_frames"] = counts
    result["stills"] = stills
    return result


def ffmpeg(*args: str) -> None:
    exe = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def encode(frames_dir: Path, out: Path, final_card: Path) -> list[Path]:
    fps = FPS // RENDER_EVERY
    seq = frames_dir / "seq"
    seq.mkdir()
    n = 0
    for name, _, _ in CLIPS:  # one numbered sequence: the clips back to back, then the ranking card held
        for f in sorted(frames_dir.glob(f"{name}_*.png")):
            os.link(f, seq / f"{n:05d}.png")
            n += 1
    for _ in range(int(HOLD_FINAL_SECONDS * fps)):
        os.link(final_card, seq / f"{n:05d}.png")
        n += 1
    mp4 = out / "match_highlights.mp4"
    ffmpeg("-framerate", str(fps), "-i", str(seq / "%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "24",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4))
    gif_seq = frames_dir / "gif"
    gif_seq.mkdir()
    gif_frames = [f for name in GIF_CLIPS for f in sorted(frames_dir.glob(f"{name}_*.png"))]
    keep = fps / GIF_FPS
    for m in range(int(len(gif_frames) / keep)):
        os.link(gif_frames[int(m * keep)], gif_seq / f"{m:05d}.png")
    gif = out / "match_highlights.gif"
    palette = frames_dir / "palette.png"
    scale = f"scale={GIF_WIDTH}:-1:flags=lanczos"
    rate = ("-framerate", str(GIF_FPS), "-i", str(gif_seq / "%05d.png"))
    ffmpeg(*rate, "-vf", f"{scale},palettegen=max_colors={GIF_COLORS}:stats_mode=diff", str(palette))
    ffmpeg(*rate, "-i", str(palette), "-lavfi",
           f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", "-loop", "0", str(gif))
    return [mp4, gif]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("replay", nargs="?", default=str(DEFAULT_REPLAY))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--no-video", action="store_true", help="stills only (no ffmpeg needed)")
    args = p.parse_args(argv)
    out = Path(args.out).resolve()
    os.chdir(ROOT)
    import pygame
    pygame.init()
    with tempfile.TemporaryDirectory(prefix="spy-vs-ai-media-") as tmp:
        frames_dir = Path(tmp)
        result = render_replay(Path(args.replay).resolve(), frames_dir, out)
        print(f"replay: {result['verdict']}, {result.get('provider_calls', 0)} provider calls, clip frames "
              f"{result['clip_frames']}")
        from spy_vs_ai.runtime import playback as pb
        if result["verdict"] != pb.REPRODUCED:
            print("the replay does not reproduce its recording: no media written")
            return 1
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name, path in result["stills"].items():
            shutil.copyfile(path, out / name)
            written.append(out / name)
        setup_still(out / "setup_screen.png")
        written.append(out / "setup_screen.png")
        if not args.no_video:
            written += encode(frames_dir, out, result["stills"]["final_ranking.png"])
    for path in written:
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}  {path.stat().st_size // 1024} KiB")
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())

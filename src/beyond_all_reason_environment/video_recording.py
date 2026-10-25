"""Record the game window into a video while a graphical engine runs (GraphicsConfiguration.video_path), from the
engine itself: the graphics widget saves each rendered frame (gl.SaveImage, after everything is drawn) into a folder
in memory (/dev/shm), and a watcher process feeds them to ffmpeg, which encodes H.264 as they come, deleting each
frame once read. Only the game's image is recorded, whatever screen or window it is on, with no question asked.

The video follows the game's time, not the clock's: one image every 30 * video_speed / video_frame_rate game frames,
played at video_frame_rate images per second (video_speed game seconds per second of video). It is smooth however
slow the machine, whatever game_speed the window runs at. Saving an image takes ~10 ms (1600x900), which slows the
window a little.

Why not a screen recorder: on GNOME Wayland with the NVIDIA card, X11 captures (ffmpeg x11grab, also of a nested
Xwayland) are black, and gpu-screen-recorder records a screen chosen by the desktop, not the game's window
(2026-10-25). gl.SaveImage replaces a named pipe by a file, hence the frames in memory.

The watcher is detached from the caller (setsid --fork): in a session of its own, out of reach of the terminal's
Ctrl+C, and no longer a descendant of the caller, whose descendants Ray kills when one of its workers ends. It ends
the video when the engine ends, however the caller ends (the engine dies with it, see match.dying_with_parent).
"""

import argparse
import glob
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .engine_configuration import GraphicsConfiguration

# Where the graphics widget saves the frames, in the run folder (the engine writes only into its write directory): a
# link to a folder in memory.
FRAMES_LINK = "video_frames"
MEMORY_DIRECTORY = Path("/dev/shm")
FRAME_NAME = "{:08d}.bmp"
GAME_FRAMES_PER_SECOND = 30
POLL_SECONDS = 0.005


def next_video_path(directory: Path, name: str) -> Path:
    """<directory>/<name>_<number>.mp4, numbered from 1 after the videos of that name already there."""
    numbers = [
        int(match.group(1))
        for path in directory.glob(f"{glob.escape(name)}_*.mp4")
        if (match := re.fullmatch(rf"{re.escape(name)}_(\d+)\.mp4", path.name))
    ]
    return directory / f"{name}_{max(numbers, default=0) + 1}.mp4"


def capture_interval(graphics: GraphicsConfiguration) -> int:
    """Game frames between two images of the video."""
    return max(1, round(GAME_FRAMES_PER_SECOND * graphics.video_speed / graphics.video_frame_rate))


def prepare_video_frames(run_directory: Path) -> Path:
    """The folder in memory for the frames, linked from the run folder (before the engine starts)."""
    memory = MEMORY_DIRECTORY if MEMORY_DIRECTORY.is_dir() else None
    frames_directory = Path(tempfile.mkdtemp(prefix="beyond_all_reason_video_", dir=memory))
    link = run_directory / FRAMES_LINK
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(frames_directory, target_is_directory=True)
    return frames_directory


def start_video_recording(
    engine: subprocess.Popen, frames_directory: Path, graphics: GraphicsConfiguration, log_path: Path
) -> None:
    """Start the watcher that encodes the frames of ``engine`` into ``graphics.video_path`` until it ends."""
    video_path = Path(graphics.video_path).resolve()  # pyright: ignore[reportArgumentType]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    setsid = shutil.which("setsid")
    detached = [setsid, "--fork"] if setsid else []
    # The rate that keeps video_speed once the interval is rounded to whole game frames.
    frame_rate = GAME_FRAMES_PER_SECOND * graphics.video_speed / capture_interval(graphics)
    with open(log_path, "w") as log_file:
        subprocess.Popen(
            [
                *detached,
                sys.executable,
                # runpy warns that the package (its __init__) imported this module before running it: harmless here.
                *("-W", "ignore::RuntimeWarning", "-m", __spec__.name),  # pyright: ignore[reportOptionalMemberAccess]
                *("--engine_process", str(engine.pid), "--frames_directory", str(frames_directory)),
                *("--video_path", str(video_path), "--frame_rate", f"{frame_rate:g}"),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def is_running(process_id: int) -> bool:
    """False once the process ended, also while it waits to be reaped by its parent (a zombie)."""
    try:
        state = Path(f"/proc/{process_id}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, IndexError):
        return False
    return state not in ("Z", "X")


def is_complete_bitmap(data: bytes) -> bool:
    """The file size written in a BMP header is the size read: not cut by an engine killed while saving."""
    return len(data) >= 6 and int.from_bytes(data[2:6], "little") == len(data)


def encode(engine_process: int, frames_directory: Path, video_path: Path, frame_rate: float) -> None:
    import imageio_ffmpeg

    encoder = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            *("-hide_banner", "-loglevel", "error", "-y"),
            *("-f", "image2pipe", "-framerate", f"{frame_rate:g}", "-c:v", "bmp", "-i", "-"),
            # H.264 in yuv420p needs even sizes.
            *("-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20"),
            *("-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    index = 0
    try:
        while True:
            frame = frames_directory / FRAME_NAME.format(index)
            following = frames_directory / FRAME_NAME.format(index + 1)
            running = is_running(engine_process)
            # A frame is whole once the next one exists, or once the engine ended (checked by its header then).
            if frame.exists() and (following.exists() or not running):
                data = frame.read_bytes()
                frame.unlink()
                if following.exists() or is_complete_bitmap(data):
                    encoder.stdin.write(data)
                index += 1
            elif not running:
                break
            else:
                time.sleep(POLL_SECONDS)
    finally:
        encoder.stdin.close()
        encoder.wait()
        shutil.rmtree(frames_directory, ignore_errors=True)
    print(f"{index} images, encoder ended (code {encoder.returncode}): {video_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode the frames of an engine into a video (see the module).")
    parser.add_argument("--engine_process", type=int, required=True)
    parser.add_argument("--frames_directory", type=Path, required=True)
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--frame_rate", type=float, required=True)
    arguments = parser.parse_args()
    encode(arguments.engine_process, arguments.frames_directory, arguments.video_path, arguments.frame_rate)

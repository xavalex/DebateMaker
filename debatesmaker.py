from __future__ import annotations

import csv
import random
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize

def format_timestamp(ms: int) -> str:
    """Formats milliseconds into ss:mmm (e.g. 12543 ms -> 12:543)."""
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{total_seconds:02d}:{milliseconds:03d}"

def get_non_overlapping_slice(audio_track, duration_ms, used_intervals):
    """Picks a random window that does not overlap previously sampled slices."""
    max_start = len(audio_track) - duration_ms
    if max_start <= 0:
        return audio_track, 0, len(audio_track)

    attempts = 0
    while attempts < 300:
        start = random.randint(0, max_start)
        end = start + duration_ms
        overlap = any(max(start, u_s) < min(end, u_e) for u_s, u_e in used_intervals)
        if not overlap:
            used_intervals.append((start, end))
            return audio_track[start:end], start, end
        attempts += 1

    # Fallback if track reaches high density
    used_intervals.append((start, end))
    return audio_track[start:end], start, end

def build_single_debate(
    selected_speakers: list[str],
    loaded_tracks: dict[str, AudioSegment],
    global_used_intervals: dict[str, list],
    target_duration_sec: int = 300,
    min_turn_sec: int = 10,
    max_turn_sec: int = 25,
    min_pause_ms: int = 0,
    max_pause_ms: int = 10000
):
    debate_track = AudioSegment.empty()
    timings = []

    target_ms = target_duration_sec * 1000
    current_ms = 0
    last_speaker = None
    turn_counter = 1

    while current_ms < target_ms:
        # Alternate strictly among the 3 chosen speakers, avoiding consecutive turns
        candidates = [s for s in selected_speakers if s != last_speaker]
        speaker = random.choice(candidates)
        last_speaker = speaker

        turn_len_ms = random.randint(min_turn_sec * 1000, max_turn_sec * 1000)
        track = loaded_tracks[speaker]

        chunk, src_start, src_end = get_non_overlapping_slice(
            track, turn_len_ms, global_used_intervals[speaker]
        )

        chunk = chunk.fade_in(60).fade_out(60)

        turn_start_ms = current_ms
        actual_chunk_len = len(chunk)
        turn_end_ms = current_ms + actual_chunk_len

        # Random silence between 0 and 10 seconds before the next turn
        pause_ms = random.randint(min_pause_ms, max_pause_ms)

        timings.append({
            "Turn": turn_counter,
            "Speaker": speaker,
            "Debate_Start": format_timestamp(turn_start_ms),
            "Debate_End": format_timestamp(turn_end_ms),
            "Duration": format_timestamp(actual_chunk_len),
            "Pause_After": format_timestamp(pause_ms),
            "Source_Window": f"{format_timestamp(src_start)} - {format_timestamp(src_end)}"
        })

        debate_track += chunk
        debate_track += AudioSegment.silent(duration=pause_ms)
        current_ms = turn_end_ms + pause_ms
        turn_counter += 1

    return debate_track, timings

def run_batch_generation(
    speaker_files: dict[str, Path],
    output_dir: Path,
    num_debates: int = 20,
    speakers_per_debate: int = 3,
    target_duration_sec: int = 300
):
    print("Loading source tracks into memory...")
    loaded_tracks = {}
    global_used_intervals = {name: [] for name in speaker_files}

    for name, path in speaker_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        print(f"  -> Ingesting {name} ({path.name})...")
        seg = AudioSegment.from_file(str(path))
        seg = seg.set_channels(1).set_frame_rate(44100)
        seg = normalize(seg)
        loaded_tracks[name] = seg

    output_dir.mkdir(parents=True, exist_ok=True)
    all_speaker_names = list(speaker_files.keys())

    print(f"\nStarting batch generation of {num_debates} debates ({speakers_per_debate} speakers each)...")
    for idx in range(1, num_debates + 1):
        audio_filename = output_dir / f"debate{idx}.mp3"
        csv_filename = output_dir / f"timing{idx}.csv"

        chosen_speakers = random.sample(all_speaker_names, speakers_per_debate)
        print(f"[{idx:02d}/{num_debates}] {audio_filename.name} -> Panel: {', '.join(chosen_speakers)}")

        debate_audio, timings = build_single_debate(
            selected_speakers=chosen_speakers,
            loaded_tracks=loaded_tracks,
            global_used_intervals=global_used_intervals,
            target_duration_sec=target_duration_sec,
            min_pause_ms=0,
            max_pause_ms=10000  # 0 to 10 seconds random pause
        )

        debate_audio.export(str(audio_filename), format="mp3", bitrate="192k")

        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Turn", "Speaker", "Debate_Start", "Debate_End", "Duration", "Pause_After", "Source_Window"
            ])
            writer.writeheader()
            writer.writerows(timings)

    print("\nBatch generation complete. All debates and CSV logs saved.")

if __name__ == "__main__":
    base_dir = Path(r"G:\My Drive\pers\Python\Logos\Speakers for training")

    speakers = {
        "Barbara Cassin": base_dir / "Cassin.mp3",
        "Alain Finkielkraut": base_dir / "Finkielkraut.mp3",
        "Jean-Marc Jancovici": base_dir / "Jancovixi.mp3",
        "Jean-Luc Mélenchon": base_dir / "Melenchon.mp3",
        "Nicolas Sarkozy": base_dir / "Sarkozy.mp3"
    }

    run_batch_generation(
        speaker_files=speakers,
        output_dir=base_dir,
        num_debates=20,
        speakers_per_debate=3,
        target_duration_sec=300
    )
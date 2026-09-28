import csv
import os
import random
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize

def format_timestamp(ms: int) -> str:
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_non_overlapping_slice(audio_track, duration_ms, used_intervals):
    """Picks a random start time that does not overlap previously sampled slices."""
    max_start = len(audio_track) - duration_ms
    if max_start <= 0:
        return audio_track, 0, len(audio_track)

    attempts = 0
    while attempts < 200:
        start = random.randint(0, max_start)
        end = start + duration_ms
        overlap = any(max(start, u_s) < min(end, u_e) for u_s, u_e in used_intervals)
        if not overlap:
            used_intervals.append((start, end))
            return audio_track[start:end], start, end
        attempts += 1

    # Fallback if density is high
    used_intervals.append((start, end))
    return audio_track[start:end], start, end

def generate_pseudo_debate(
    speaker_files: dict,
    output_audio_path: Path,
    output_csv_path: Path,
    target_duration_sec: int = 300,
    min_turn_sec: int = 10,
    max_turn_sec: int = 25,
    pause_ms: int = 350
):
    print("Loading and normalizing audio tracks...")
    loaded_tracks = {}
    used_intervals = {name: [] for name in speaker_files}

    for name, path in speaker_files.items():
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        print(f"  -> Ingesting {name} ({path.name})...")
        seg = AudioSegment.from_file(str(path))
        # Standardize to 44.1kHz Mono and normalize volume
        seg = seg.set_channels(1).set_frame_rate(44100)
        seg = normalize(seg)
        loaded_tracks[name] = seg

    speaker_names = list(speaker_files.keys())
    debate_track = AudioSegment.empty()
    timings = []

    target_ms = target_duration_sec * 1000
    current_ms = 0
    last_speaker = None
    turn_counter = 1

    print("\nAssembling debate turns...")
    while current_ms < target_ms:
        candidates = [s for s in speaker_names if s != last_speaker]
        speaker = random.choice(candidates)
        last_speaker = speaker

        turn_len_ms = random.randint(min_turn_sec * 1000, max_turn_sec * 1000)
        track = loaded_tracks[speaker]

        chunk, src_start, src_end = get_non_overlapping_slice(
            track, turn_len_ms, used_intervals[speaker]
        )

        # 60ms micro-fade to eliminate clicks at cut boundaries
        chunk = chunk.fade_in(60).fade_out(60)

        turn_start_ms = current_ms
        turn_end_ms = current_ms + len(chunk)

        timings.append({
            "Turn": turn_counter,
            "Speaker": speaker,
            "Debate_Start": format_timestamp(turn_start_ms),
            "Debate_End": format_timestamp(turn_end_ms),
            "Duration_s": round(len(chunk) / 1000, 1),
            "Source_Window": f"{format_timestamp(src_start)} - {format_timestamp(src_end)}"
        })

        debate_track += chunk
        debate_track += AudioSegment.silent(duration=pause_ms)
        current_ms = turn_end_ms + pause_ms
        turn_counter += 1

    print(f"\nExporting master audio to: {output_audio_path}")
    debate_track.export(str(output_audio_path), format="mp3", bitrate="192k")

    print(f"Writing timing log to: {output_csv_path}")
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Turn", "Speaker", "Debate_Start", "Debate_End", "Duration_s", "Source_Window"
        ])
        writer.writeheader()
        writer.writerows(timings)

    print("\n--- Pseudo-Debate Timings ---")
    header = f"{'Turn':<6} {'Speaker':<20} {'Start':<8} {'End':<8} {'Duration':<10} {'Source Offset'}"
    print(header)
    print("-" * len(header))
    for t in timings:
        print(f"{t['Turn']:<6} {t['Speaker']:<20} {t['Debate_Start']:<8} {t['Debate_End']:<8} {str(t['Duration_s']) + 's':<10} {t['Source_Window']}")

if __name__ == "__main__":
    base_dir = Path(r"G:\My Drive\pers\Python\Logos\Speakers for training")

    speakers = {
        "Barbara Cassin": base_dir / "Cassin.mp3",
        "Alain Finkielkraut": base_dir / "Finkielkraut.mp3",
        "Jean-Marc Jancovici": base_dir / "Jancovixi.mp3",
        "Jean-Luc Mélenchon": base_dir / "Melenchon.mp3",
        "Nicolas Sarkozy": base_dir / "Sarkozy.mp3"
    }

    out_audio = base_dir / "pseudo_debate_5min.mp3"
    out_csv = base_dir / "debate_timings.csv"

    generate_pseudo_debate(
        speaker_files=speakers,
        output_audio_path=out_audio,
        output_csv_path=out_csv,
        target_duration_sec=300,  # 5 minutes
        min_turn_sec=10,          # 10s min per turn
        max_turn_sec=22,          # 22s max per turn
        pause_ms=350              # Natural pause between turns
    )
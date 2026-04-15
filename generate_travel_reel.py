"""
Travel Destination Reel Generator
- Accepts destination name as command-line argument
- Generates Hindi voiceover covering top attractions using Gemini TTS (Erinome)
- Fetches unique Pexels video clips for each segment (with disk cache)
- Builds 9:16 reel with destination overlay + background music
- Sends final video to Telegram chat
- Falls back to ChatGPT if Gemini text generation fails
"""

import os
import sys
import re
import time
import requests
import random
import json
import shutil
import subprocess
import openai
from google import genai
from google.genai import types
from pathlib import Path
from datetime import datetime
from moviepy.editor import (
    VideoFileClip, ColorClip, CompositeVideoClip,
    AudioFileClip, concatenate_videoclips, ImageClip,
    CompositeAudioClip
)
from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.video.fx.all import crop, resize
from moviepy.audio.fx.all import volumex

# FIX: Patch removed Pillow constant for MoviePy 1.0.3 compatibility
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- Constants ---
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PEXELS_API_KEY = os.environ["PEXELS_API_KEY"]
REEL_W, REEL_H = 1080, 1920

# Pexels video disk cache — survives across workflow steps within the same job
CACHE_DIR = Path(os.environ.get("PEXELS_CACHE_DIR", "/tmp/pexels_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Background music: place a royalty-free .mp3 named bg_music.mp3 in your repo root
BG_MUSIC_PATH = os.environ.get("BG_MUSIC_PATH", "bg_music.mp3")
BG_MUSIC_VOLUME = 0.12   # 12% — keeps voiceover dominant
VOICEOVER_SPEED = 1.10   # 10% faster — natural-sounding, saves ~9s per 100s

# Read destination from command line or env var
if len(sys.argv) > 1:
    DESTINATION = sys.argv[1]
else:
    DESTINATION = os.environ.get("DESTINATION", "")
    if not DESTINATION:
        print("ERROR: Provide destination as argument or DESTINATION env var.")
        sys.exit(1)

# --- Telegram Sender ---
def send_video_telegram(video_path: str, caption: str = ""):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[ERROR] Missing Telegram credentials")
        return False
    url = f"https://api.telegram.org/bot{token}/sendVideo"
    try:
        with open(video_path, "rb") as f:
            response = requests.post(
                url,
                files={"video": f},
                data={"chat_id": chat_id, "caption": caption},
                timeout=120
            )
        response.raise_for_status()
        print("[SUCCESS] Video sent to Telegram")
        return True
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")
        return False

# --- Script Generation ---
def _build_prompt(destination: str) -> str:
    return (
        "Aap ek premium travel company 'Sky Suffer Tourism Private Limited' ke liye Instagram Reels scriptwriter hain.\n"
        f"Aaj hum destination \"{destination}\" par ek exciting reel banayenge.\n\n"
        f"Ek Hindi voiceover script likho jo {destination} ke top tourist spots, unki khoobsurti, aur wahan jaane ke reasons ko highlight kare.\n"
        "Script exciting aur inviting tone mein honi chahiye, jaise ek tour guide apne guests ko roam karwa raha ho.\n\n"
        "IMPORTANT: Script ke saath Pexels video search keywords bhi provide karo — har location ya vibe ke liye alag keyword.\n"
        "Sirf valid JSON return karo is exact format mein:\n"
        '{{"script": "Poora voiceover script yahan...", "segments": [{{"description": "Is part mein kya dikhaya ja raha hai", "keywords": "pexels search term in English"}}, ...]}}\n\n'
        "Script rules:\n"
        f"- Shuru karo: 'Sky Suffer Tourism ke saath aaj hum aapko le ja rahe hain {destination} ki yaadgaar yatra par!'\n"
        f"- {destination} ke 4-6 sabse famous aur beautiful tourist spots ko cover karo.\n"
        "- Har spot ke baare mein 2-3 sentences bolo — uska khaas attraction kya hai, kyun visit karna chahiye.\n"
        "- Exciting adjectives use karo: 'shaandar', 'manmohak', 'adbhut', 'swarg se kam nahi'.\n"
        f"- Ant mein kaho: 'Toh der kis baat ki? Abhi contact karein Sky Suffer Tourism Private Limited ko 9654100207 par aur book karein apna dream trip {destination} ka!'\n"
        "- Total script 200-250 words. Natural, flowing Hindi.\n\n"
        "Keywords rules:\n"
        "- Har tourist spot ya visual vibe ke liye ek segment banao.\n"
        "- Keywords ENGLISH mein, simple 1-3 word Pexels search terms."
    )

def _parse_response(text: str) -> dict:
    """Robustly extract JSON from a response that may contain markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)

def generate_script(destination: str) -> dict:
    prompt = _build_prompt(destination)
    gemini_models = ["gemini-2.0-flash", "gemini-2.5-flash"]

    for model in gemini_models:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"[INFO] Trying Gemini model: {model}")
            response = client.models.generate_content(model=model, contents=prompt)
            data = _parse_response(response.text)
            print(f"[INFO] Script generated using Gemini {model} ({len(data['script'].split())} words, {len(data['segments'])} segments)")
            return data
        except Exception as e:
            print(f"[WARN] Gemini {model} failed: {e}")

    if not OPENAI_API_KEY:
        raise RuntimeError("All Gemini models failed and OPENAI_API_KEY not set.")
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON in the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        response_text = response.choices[0].message.content
        data = _parse_response(response_text)
        print(f"[INFO] Script generated using ChatGPT ({len(data['script'].split())} words, {len(data['segments'])} segments)")
        return data
    except Exception as e:
        print(f"[ERROR] ChatGPT also failed: {e}")
        raise RuntimeError("Both Gemini and ChatGPT failed to generate script.")

# --- Pexels Video Fetch (disk cache + deduplication + robust frame validation) ---
used_video_ids = set()

def _cache_filename(video_id: int, width: int) -> Path:
    return CACHE_DIR / f"{video_id}_{width}.mp4"

def _is_valid_clip(path: str) -> bool:
    """
    FIX: Open the clip AND decode frame 0 — catches broken/truncated containers
    that VideoFileClip() alone won't reject.
    """
    try:
        clip = VideoFileClip(path)
        clip.get_frame(0)
        clip.close()
        return True
    except Exception as e:
        print(f"[WARN] Clip validation failed for {path}: {e}")
        return False

def fetch_pexels_video(query: str, output_path: str, max_retries: int = 2):
    headers = {"Authorization": PEXELS_API_KEY}

    videos = []
    for orientation in ["portrait", "landscape"]:
        params = {"query": query, "per_page": 15, "orientation": orientation, "size": "medium"}
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(
                    "https://api.pexels.com/videos/search",
                    headers=headers, params=params, timeout=15
                )
                resp.raise_for_status()
                videos = resp.json().get("videos", [])
                break
            except requests.HTTPError as e:
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"[WARN] Pexels rate limit hit, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[WARN] Pexels HTTP error {resp.status_code} for '{query}': {e}")
                    break
            except Exception as e:
                print(f"[WARN] Pexels request failed for '{query}': {e}")
                break
        if videos:
            break

    if not videos:
        print(f"[WARN] No videos found for '{query}'")
        return None

    available = [v for v in videos if v["id"] not in used_video_ids]
    if not available:
        available = videos

    candidates = available[:5]
    random.shuffle(candidates)

    for video in candidates:
        used_video_ids.add(video["id"])
        video_files = sorted(video["video_files"], key=lambda x: x.get("width", 0), reverse=True)
        chosen = next((v for v in video_files if v.get("width", 0) <= 1920), video_files[0])
        chosen_width = chosen.get("width", 0)

        # --- Check disk cache first ---
        cache_file = _cache_filename(video["id"], chosen_width)
        if cache_file.exists():
            if _is_valid_clip(str(cache_file)):
                print(f"[CACHE HIT] {query} -> {cache_file.name}")
                shutil.copy2(str(cache_file), output_path)
                return output_path
            else:
                cache_file.unlink(missing_ok=True)  # stale cache entry

        # --- Download ---
        print(f"[INFO] Downloading: {query} -> {chosen['link'][:60]}... (ID: {video['id']})")
        try:
            r = requests.get(chosen["link"], timeout=60, stream=True)
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"[WARN] Download failed for video {video['id']}: {e}")
            continue

        # --- Validate with frame decode before using ---
        if not _is_valid_clip(output_path):
            os.remove(output_path)
            print(f"[WARN] Video {video['id']} failed frame validation, trying next candidate...")
            continue

        # --- Save to disk cache ---
        shutil.copy2(output_path, str(cache_file))
        print(f"[CACHE SAVE] {cache_file.name}")
        return output_path

    print(f"[WARN] All candidates failed for '{query}'")
    return None

# --- Voiceover Generation (Gemini TTS - Erinome, sped up) ---
def generate_voiceover(script: str, output_path: str = "voiceover.mp3") -> str:
    import imageio_ffmpeg

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=script,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Erinome"
                    )
                )
            )
        )
    )

    audio_data = b""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            audio_data += part.inline_data.data

    raw_path = output_path.replace(".mp3", ".raw")
    wav_path = output_path.replace(".mp3", ".wav")
    sped_wav_path = output_path.replace(".mp3", "_sped.wav")

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    with open(raw_path, "wb") as f:
        f.write(audio_data)

    # Raw PCM -> WAV
    subprocess.run([
        ffmpeg_bin, "-y",
        "-f", "s16le", "-ar", "24000", "-ac", "1",
        "-i", raw_path, wav_path
    ], check=True, capture_output=True)

    # Speed up using atempo filter
    subprocess.run([
        ffmpeg_bin, "-y", "-i", wav_path,
        "-filter:a", f"atempo={VOICEOVER_SPEED}",
        sped_wav_path
    ], check=True, capture_output=True)

    # Sped WAV -> MP3
    subprocess.run([
        ffmpeg_bin, "-y", "-i", sped_wav_path, output_path
    ], check=True, capture_output=True)

    for p in [raw_path, wav_path, sped_wav_path]:
        if os.path.exists(p):
            os.remove(p)

    print(f"[INFO] Gemini TTS voiceover (Erinome, {VOICEOVER_SPEED}x) saved -> {output_path}")
    return output_path

# --- Background Music ---
def load_bg_music(duration: float):
    """Load bg_music.mp3, loop to fill duration, return quietened clip or None."""
    if not os.path.exists(BG_MUSIC_PATH):
        print(f"[WARN] No background music at '{BG_MUSIC_PATH}', skipping.")
        return None
    try:
        music = AudioFileClip(BG_MUSIC_PATH)
        if music.duration < duration:
            loops = int(duration / music.duration) + 1
            music = concatenate_audioclips([music] * loops)
        music = music.subclip(0, duration)
        music = volumex(music, BG_MUSIC_VOLUME)
        print(f"[INFO] BG music loaded, looped to {duration:.1f}s at {int(BG_MUSIC_VOLUME*100)}% volume")
        return music
    except Exception as e:
        print(f"[WARN] Could not load background music: {e}")
        return None

# --- Video Assembly ---
def make_text_image(text, fontsize, color, stroke_color="black", stroke_width=3):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + stroke_width * 2 + 20
    h = bbox[3] - bbox[1] + stroke_width * 2 + 20
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            draw.text((stroke_width + 10 + dx, stroke_width + 10 + dy), text, font=font, fill=stroke_color)
    draw.text((stroke_width + 10, stroke_width + 10), text, font=font, fill=color)
    return np.array(img)

def make_destination_overlay(destination: str, duration: float):
    dest_text = make_text_image(destination, 80, "#FFC800")
    img_clip = ImageClip(dest_text)
    img_w = dest_text.shape[1]
    x_pos = (REEL_W - img_w) // 2
    y_pos = int(REEL_H * 0.25)
    dest_clip = img_clip.set_duration(duration).set_position((x_pos, y_pos))
    return [dest_clip]

def crop_to_portrait(clip):
    target_ratio = REEL_W / REEL_H
    clip_ratio = clip.w / clip.h
    if clip_ratio > target_ratio:
        new_w = int(clip.h * target_ratio)
        clip = crop(clip, width=new_w, x_center=clip.w / 2)
    else:
        new_h = int(clip.w / target_ratio)
        clip = crop(clip, height=new_h, y_center=clip.h / 2)
    return resize(clip, (REEL_W, REEL_H))

def build_video(data: dict, audio_path: str, destination: str, output_path: str = "reel.mp4") -> str:
    voiceover = AudioFileClip(audio_path)
    duration = voiceover.duration
    segments = data["segments"]
    n = len(segments)
    seg_duration = duration / n
    print(f"[INFO] Video duration: {duration:.1f}s, {n} segments x {seg_duration:.1f}s each")

    clip_paths = []
    clips = []

    for i, seg in enumerate(segments):
        clip_path = f"clip_{i}.mp4"
        clip_paths.append(clip_path)

        pexels_path = fetch_pexels_video(seg["keywords"], clip_path)
        if not pexels_path:
            fallback_keyword = seg["keywords"].split(",")[0].split()[0]
            print(f"[INFO] Retrying segment {i} with fallback keyword: '{fallback_keyword}'")
            pexels_path = fetch_pexels_video(fallback_keyword, clip_path)

        raw = None
        if pexels_path:
            try:
                raw = VideoFileClip(pexels_path)
                raw.get_frame(0)  # double-check inside build too
                if raw.duration < seg_duration:
                    loops = int(seg_duration / raw.duration) + 1
                    looped = concatenate_videoclips([raw] * loops)
                    raw.close()
                    raw = looped
                raw = raw.subclip(0, seg_duration)
                raw = crop_to_portrait(raw).without_audio()
            except Exception as e:
                print(f"[WARN] Clip {i} failed in build: {e}, using color fallback")
                try:
                    if raw:
                        raw.close()
                except Exception:
                    pass
                raw = ColorClip(size=(REEL_W, REEL_H), color=(15, 15, 30)).set_duration(seg_duration)
        else:
            raw = ColorClip(size=(REEL_W, REEL_H), color=(15, 15, 30)).set_duration(seg_duration)

        overlay = ColorClip(size=(REEL_W, REEL_H), color=(0, 0, 0)).set_opacity(0.45).set_duration(seg_duration)
        clips.append(CompositeVideoClip([raw, overlay], size=(REEL_W, REEL_H)))

    base_video = concatenate_videoclips(clips, method="compose")
    overlays = make_destination_overlay(destination, duration)

    # Mix voiceover + background music
    bg_music = load_bg_music(duration)
    final_audio = CompositeAudioClip([voiceover, bg_music]) if bg_music else voiceover

    final = (
        CompositeVideoClip([base_video] + overlays, size=(REEL_W, REEL_H))
        .set_audio(final_audio)
        .set_duration(duration)
    )
    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        temp_audiofile="temp_audio.m4a", remove_temp=True, logger=None,
        ffmpeg_params=[
            "-profile:v", "baseline", "-level", "3.0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-b:v", "3500k", "-b:a", "128k", "-ar", "44100", "-ac", "2"
        ]
    )

    voiceover.close()
    if bg_music:
        bg_music.close()
    final.close()
    base_video.close()
    for clip in clips:
        try:
            clip.close()
        except Exception:
            pass

    for path in clip_paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Cleaned up temp clip: {path}")

    print(f"[INFO] Video built -> {output_path}")
    return output_path

# --- Cleanup helper ---
def cleanup_temp_files(*paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"[INFO] Cleaned up: {path}")
            except Exception as e:
                print(f"[WARN] Could not delete {path}: {e}")

# --- Main ---
def main():
    import imageio_ffmpeg
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

    print("=" * 60)
    print(f"Travel Reel Generator — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Destination: {DESTINATION}")
    print("=" * 60)

    audio_path = "voiceover.mp3"
    video_path = "reel.mp4"

    try:
        data = generate_script(DESTINATION)
        generate_voiceover(data["script"], audio_path)
        build_video(data, audio_path, DESTINATION, video_path)

        caption = (
            f"✨ {DESTINATION} awaits you with Sky Suffer Tourism! ✨\n\n"
            f"Discover the magic of {DESTINATION} — from breathtaking landscapes to rich culture.\n\n"
            f"📞 Contact us: 9654100207\n"
            f"📌 Save this reel and start planning your next adventure!\n\n"
            f"#SkySafarTourism #{DESTINATION.replace(' ', '')} #Travel #Wanderlust"
        )
        send_video_telegram(video_path, caption)
        print("\n✅ Done! Reel sent to Telegram.")
    finally:
        cleanup_temp_files(audio_path, video_path)

if __name__ == "__main__":
    main()

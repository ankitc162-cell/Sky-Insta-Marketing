"""
Travel Destination Reel Generator
- Accepts destination name + optional audience type as CLI args
- Generates story-driven Hindi voiceover with scroll-stopping hook via Gemini TTS (Erinome)
- Fetches Pexels video clips with cinematic keyword hints (disk cached)
- Builds 9:16 reel with semi-transparent destination overlay + company logo at end
- Mixes viral-style background music at 15% volume
- Sends final video to Telegram
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
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
PEXELS_API_KEY  = os.environ["PEXELS_API_KEY"]
REEL_W, REEL_H  = 1080, 1920

CACHE_DIR       = Path(os.environ.get("PEXELS_CACHE_DIR", "/tmp/pexels_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BG_MUSIC_PATH   = os.environ.get("BG_MUSIC_PATH", "bg_music.mp3")
BG_MUSIC_VOLUME = 0.15          # 15% volume
VOICEOVER_SPEED = 1.10          # 10% faster
LOGO_PATH       = os.environ.get("LOGO_PATH", "logo.png")  # company logo file

# Pixabay viral music fallback URLs (royalty-free, no attribution required)
VIRAL_MUSIC_OPTIONS = [
    "https://cdn.pixabay.com/download/audio/2024/03/13/audio_3d5b6e6e21.mp3",  # Epic Cinematic
    "https://cdn.pixabay.com/download/audio/2023/06/05/audio_0bbfe02b94.mp3",  # Travel Vlog
    "https://cdn.pixabay.com/download/audio/2022/10/25/audio_946f989511.mp3",  # Inspirational
]

# Read destination + optional audience from CLI args or env vars
if len(sys.argv) > 1:
    DESTINATION   = sys.argv[1]
    AUDIENCE_TYPE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("AUDIENCE_TYPE", "general travelers")
else:
    DESTINATION   = os.environ.get("DESTINATION", "")
    AUDIENCE_TYPE = os.environ.get("AUDIENCE_TYPE", "general travelers")
    if not DESTINATION:
        print("ERROR: Provide destination as argument or DESTINATION env var.")
        sys.exit(1)

# --- Telegram Sender ---
def send_video_telegram(video_path: str, caption: str = ""):
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
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
def _build_prompt(destination: str, audience_type: str) -> str:
    return (
        "Aap ek premium travel company 'Sky Suffer Tourism Private Limited' ke liye viral Instagram Reels scriptwriter hain.\n"
        f"Destination: \"{destination}\"\n"
        f"Target audience: {audience_type} (tone aur language isi ke hisaab se adjust karo)\n\n"

        "Sirf valid JSON return karo — koi preamble nahi, koi markdown fence nahi — is exact format mein:\n"
        "{\n"
        '  "hook": "First 3-second scroll-stopping line",\n'
        '  "viral_line": "One deeply emotional/quotable line from the script",\n'
        '  "music_vibe": "cinematic / upbeat / romantic — whichever fits best",\n'
        '  "script": "Complete voiceover script from hook to CTA",\n'
        '  "segments": [\n'
        '    {"description": "What is shown", "keywords": "cinematic drone sunset beach", "mood": "golden hour peaceful"}\n'
        '  ]\n'
        "}\n\n"

        "=== SCRIPT RULES ===\n"
        f"HOOK (first 3 seconds — curiosity/FOMO/shock, NOT brand-first):\n"
        f"  - Ek aisa sawaal ya statement jo viewer ko ruk jaane par majboor kare\n"
        f"  - Example style: 'Kya aapne kabhi aisi jagah dekhi hai jahan badal zameen ko chho lete hain?'\n"
        f"  - THEN: 'Sky Suffer Tourism ke saath aaj hum aapko le ja rahe hain {destination} ki yaadgaar yatra par!'\n\n"

        "STORYTELLING (NOT a brochure — paint moments):\n"
        f"  - {destination} ke 4-6 sabse iconic spots cover karo\n"
        "  - Har spot ke liye ek micro-moment banao. Example:\n"
        "    BAD:  'Yeh jagah bahut sundar hai'\n"
        "    GOOD: 'Subah jab suraj ki pehli kiran in chaaon mein padti hai, toh waqt ruk jaata hai'\n"
        "  - Audience-aware tone:\n"
        "    * Couples/honeymoon → romantic, intimate\n"
        "    * Friends → fun, adventurous, FOMO-heavy\n"
        "    * Luxury → exclusivity, world-class, rare experience\n"
        "    * General → wonder, wanderlust, bucket-list\n\n"

        "VIRAL LINE:\n"
        "  - Ek line jo itni powerful ho ki log screenshot lein ya share karein\n"
        "  - Example: 'Yeh sirf ek jagah nahi, ek ehsaas hai jo zindagi bhar yaad rahega'\n\n"

        "CTA (urgent + emotional, NOT just functional):\n"
        "  - Pehle urgency/FOMO: 'offers limited hain — aur aisi jagah baar-baar nahi milti...'\n"
        f"  - Phir: 'Abhi contact karein Sky Suffer Tourism Private Limited ko 9654100207 par aur book karein apna dream trip {destination} ka!'\n\n"

        "  - Total script 160-180 words. Natural, flowing Hindi. Hook included in word count.\n\n"

        "=== KEYWORDS RULES ===\n"
        "  - Har segment ke liye cinematic-style 3-5 word Pexels search terms\n"
        "  - Include camera style + mood: 'drone mountain sunrise mist', 'slow motion waterfall forest'\n"
        "  - Keywords ENGLISH mein only\n\n"

        "=== MUSIC VIBE ===\n"
        "  - Choose ONE: cinematic / upbeat / romantic\n"
        f"  - Based on destination + audience: {audience_type} visiting {destination}"
    )

def _parse_response(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)

def generate_script(destination: str, audience_type: str) -> dict:
    prompt = _build_prompt(destination, audience_type)
    gemini_models = ["gemini-2.0-flash", "gemini-2.5-flash"]

    for model in gemini_models:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"[INFO] Trying Gemini model: {model}")
            response = client.models.generate_content(model=model, contents=prompt)
            data = _parse_response(response.text)
            print(f"[INFO] Script generated via Gemini {model} | {len(data['script'].split())} words | {len(data['segments'])} segments")
            print(f"[INFO] Hook: {data.get('hook', 'N/A')[:80]}...")
            print(f"[INFO] Viral line: {data.get('viral_line', 'N/A')[:80]}...")
            print(f"[INFO] Music vibe: {data.get('music_vibe', 'N/A')}")
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
                {"role": "system", "content": "You are a helpful assistant. Output only valid JSON, no markdown, no preamble."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        data = _parse_response(response.choices[0].message.content)
        print(f"[INFO] Script generated via ChatGPT | {len(data['script'].split())} words | {len(data['segments'])} segments")
        return data
    except Exception as e:
        print(f"[ERROR] ChatGPT also failed: {e}")
        raise RuntimeError("Both Gemini and ChatGPT failed to generate script.")

# --- Background Music ---
def get_bg_music_path() -> str:
    """Use local bg_music.mp3 if present, otherwise download a viral royalty-free track."""
    if os.path.exists(BG_MUSIC_PATH):
        print(f"[INFO] Using local background music: {BG_MUSIC_PATH}")
        return BG_MUSIC_PATH

    print("[INFO] No local bg_music.mp3 found — downloading viral royalty-free track...")
    for url in VIRAL_MUSIC_OPTIONS:
        try:
            r = requests.get(url, timeout=30, stream=True)
            r.raise_for_status()
            tmp_path = "/tmp/bg_music_dl.mp3"
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[INFO] Downloaded background music from Pixabay: {url[:60]}...")
            return tmp_path
        except Exception as e:
            print(f"[WARN] Failed to download music from {url[:60]}: {e}")

    print("[WARN] All music download attempts failed — proceeding without background music.")
    return None

def load_bg_music(duration: float):
    music_path = get_bg_music_path()
    if not music_path:
        return None
    try:
        music = AudioFileClip(music_path)
        if music.duration < duration:
            loops = int(duration / music.duration) + 1
            music = concatenate_audioclips([music] * loops)
        music = music.subclip(0, duration)
        music = volumex(music, BG_MUSIC_VOLUME)
        print(f"[INFO] BG music ready: {duration:.1f}s at {int(BG_MUSIC_VOLUME*100)}% volume")
        return music
    except Exception as e:
        print(f"[WARN] Could not load background music: {e}")
        return None

# --- Pexels Video Fetch (with validation and retry) ---
used_video_ids = set()

def _cache_filename(video_id: int, width: int) -> Path:
    return CACHE_DIR / f"{video_id}_{width}.mp4"

def _is_valid_video_file(path: str) -> bool:
    """Check if video file is playable using ffprobe."""
    try:
        import imageio_ffmpeg
        ffprobe_bin = imageio_ffmpeg.get_ffmpeg_exe().replace("ffmpeg", "ffprobe")
        result = subprocess.run(
            [ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10
        )
        return "video" in result.stdout
    except Exception:
        return False

def fetch_pexels_video(query: str, output_path: str, max_retries: int = 3):
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
                    print(f"[WARN] Pexels rate limit, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[WARN] Pexels HTTP {resp.status_code} for '{query}': {e}")
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
        print(f"[INFO] All videos for '{query}' already used; allowing reuse.")

    candidates = available[:8]
    random.shuffle(candidates)

    for video in candidates:
        used_video_ids.add(video["id"])
        video_files = sorted(video["video_files"], key=lambda x: x.get("width", 0), reverse=True)
        chosen = next((v for v in video_files if v.get("width", 0) <= 1920), video_files[0])
        chosen_width = chosen.get("width", 0)

        cache_file = _cache_filename(video["id"], chosen_width)
        if cache_file.exists():
            if _is_valid_video_file(str(cache_file)):
                print(f"[CACHE HIT] {query} -> {cache_file.name}")
                shutil.copy2(str(cache_file), output_path)
                return output_path
            else:
                cache_file.unlink(missing_ok=True)

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

        if not _is_valid_video_file(output_path):
            os.remove(output_path)
            print(f"[WARN] Video {video['id']} is corrupt, trying next candidate...")
            continue

        shutil.copy2(output_path, str(cache_file))
        print(f"[CACHE SAVE] {cache_file.name}")
        return output_path

    print(f"[ERROR] All candidate videos for '{query}' failed. Returning None.")
    return None

# --- Voiceover Generation ---
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
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Erinome")
                )
            )
        )
    )

    audio_data = b""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            audio_data += part.inline_data.data

    raw_path  = output_path.replace(".mp3", ".raw")
    wav_path  = output_path.replace(".mp3", ".wav")
    sped_path = output_path.replace(".mp3", "_sped.wav")
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    with open(raw_path, "wb") as f:
        f.write(audio_data)

    subprocess.run([ffmpeg_bin, "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-i", raw_path, wav_path], check=True, capture_output=True)
    subprocess.run([ffmpeg_bin, "-y", "-i", wav_path,
                    "-filter:a", f"atempo={VOICEOVER_SPEED}", sped_path],
                   check=True, capture_output=True)
    subprocess.run([ffmpeg_bin, "-y", "-i", sped_path, output_path],
                   check=True, capture_output=True)

    for p in [raw_path, wav_path, sped_path]:
        if os.path.exists(p):
            os.remove(p)

    print(f"[INFO] Voiceover saved -> {output_path} ({VOICEOVER_SPEED}x speed)")
    return output_path

# --- Text / Image Helpers ---
def make_text_image(text, fontsize, color, stroke_color="black", stroke_width=3, alpha=255):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    dummy = Image.new("RGBA", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + stroke_width * 2 + 20
    h = bbox[3] - bbox[1] + stroke_width * 2 + 20
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if isinstance(color, str) and color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fill_color = (r, g, b, alpha)
    else:
        fill_color = color

    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            draw.text((stroke_width + 10 + dx, stroke_width + 10 + dy),
                      text, font=font, fill=stroke_color)
    draw.text((stroke_width + 10, stroke_width + 10), text, font=font, fill=fill_color)
    return np.array(img)

def make_destination_overlay(destination: str, duration: float):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 75)
            break
    if font is None:
        font = ImageFont.load_default()

    spaced_text = " ".join(destination.upper())

    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), spaced_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad = 30
    img = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill_color = (255, 255, 255, 179)
    stroke_color = (200, 200, 200, 179)
    stroke_width = 2

    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            draw.text((pad + dx, pad + dy), spaced_text, font=font, fill=stroke_color)
    draw.text((pad, pad), spaced_text, font=font, fill=fill_color)

    y_pos = int(REEL_H * 0.10)
    x_pos = (REEL_W - img.width) // 2

    img_clip = ImageClip(np.array(img))
    return [img_clip.set_duration(duration).set_position((x_pos, y_pos))]

def make_logo_overlay(duration: float):
    if not os.path.exists(LOGO_PATH):
        print(f"[WARN] Logo not found at '{LOGO_PATH}', skipping.")
        return []
    try:
        from PIL import Image
        import numpy as np
        logo_img = Image.open(LOGO_PATH).convert("RGBA")
        max_h = 100
        ratio = max_h / logo_img.height
        new_w = int(logo_img.width * ratio)
        logo_img = logo_img.resize((new_w, max_h), PIL.Image.LANCZOS)
        logo_arr = np.array(logo_img)

        logo_clip = ImageClip(logo_arr)
        x_pos = (REEL_W - new_w) // 2
        y_pos = REEL_H - max_h - 40

        logo_clip = logo_clip.set_duration(duration).set_position((x_pos, y_pos))
        print(f"[INFO] Logo overlay: centered bottom, entire duration")
        return [logo_clip]
    except Exception as e:
        print(f"[WARN] Could not load logo: {e}")
        return []

def crop_to_portrait(clip):
    target_ratio = REEL_W / REEL_H
    clip_ratio   = clip.w / clip.h
    if clip_ratio > target_ratio:
        new_w = int(clip.h * target_ratio)
        clip  = crop(clip, width=new_w, x_center=clip.w / 2)
    else:
        new_h = int(clip.w / target_ratio)
        clip  = crop(clip, height=new_h, y_center=clip.h / 2)
    return resize(clip, (REEL_W, REEL_H))

# --- Video Assembly ---
def build_video(data: dict, audio_path: str, destination: str, output_path: str = "reel.mp4") -> str:
    voiceover    = AudioFileClip(audio_path)
    duration     = voiceover.duration
    segments     = data["segments"]
    n            = len(segments)
    seg_duration = duration / n
    print(f"[INFO] Duration: {duration:.1f}s | {n} segments x {seg_duration:.1f}s each")

    clip_paths = []
    clips      = []

    for i, seg in enumerate(segments):
        clip_path = f"clip_{i}.mp4"
        clip_paths.append(clip_path)

        pexels_path = fetch_pexels_video(seg["keywords"], clip_path)

        if not pexels_path:
            spot_hint = seg.get("description", "").split()[0] if seg.get("description") else ""
            fallback1 = f"{destination} {spot_hint}".strip()
            print(f"[INFO] Segment {i} primary failed. Trying: '{fallback1}'")
            pexels_path = fetch_pexels_video(fallback1, clip_path)

        if not pexels_path:
            fallback2 = f"{destination} travel"
            print(f"[INFO] Still no clip. Trying: '{fallback2}'")
            pexels_path = fetch_pexels_video(fallback2, clip_path)

        raw = None
        if pexels_path:
            try:
                raw = VideoFileClip(pexels_path)
                raw.get_frame(0)
                if raw.duration < seg_duration:
                    loops = int(seg_duration / raw.duration) + 1
                    raw = concatenate_videoclips([raw] * loops)
                raw = raw.subclip(0, seg_duration)
                raw = crop_to_portrait(raw).without_audio()
            except Exception as e:
                print(f"[WARN] Clip {i} processing failed: {e}, using color fallback")
                try:
                    if raw:
                        raw.close()
                except:
                    pass
                raw = ColorClip(size=(REEL_W, REEL_H), color=(15, 15, 30)).set_duration(seg_duration)
        else:
            print(f"[WARN] All destination‑specific queries failed for segment {i}. Using color clip.")
            raw = ColorClip(size=(REEL_W, REEL_H), color=(15, 15, 30)).set_duration(seg_duration)

        overlay = ColorClip(size=(REEL_W, REEL_H), color=(0, 0, 0)).set_opacity(0.45).set_duration(seg_duration)
        clips.append(CompositeVideoClip([raw, overlay], size=(REEL_W, REEL_H)))

    base_video = concatenate_videoclips(clips, method="compose")

    dest_overlays = make_destination_overlay(destination, duration)
    logo_overlays = make_logo_overlay(duration)
    all_overlays  = dest_overlays + logo_overlays

    bg_music    = load_bg_music(duration)
    final_audio = CompositeAudioClip([voiceover, bg_music]) if bg_music else voiceover

    final = (
        CompositeVideoClip([base_video] + all_overlays, size=(REEL_W, REEL_H))
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
        except:
            pass
    for path in clip_paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Cleaned up temp clip: {path}")

    print(f"[INFO] Video built -> {output_path}")
    return output_path

# --- Cleanup ---
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
    print(f"Destination:   {DESTINATION}")
    print(f"Audience:      {AUDIENCE_TYPE}")
    print("=" * 60)

    audio_path = "voiceover.mp3"
    video_path = "reel.mp4"

    try:
        data = generate_script(DESTINATION, AUDIENCE_TYPE)
        generate_voiceover(data["script"], audio_path)
        build_video(data, audio_path, DESTINATION, video_path)

        caption = (
            f"✨ {DESTINATION} awaits you with Sky Suffer Tourism! ✨\n\n"
            f"{data.get('viral_line', '')}\n\n"
            f"Discover the magic of {DESTINATION} — from breathtaking landscapes to rich culture.\n\n"
            f"📞 Contact us: 9654100207\n"
            f"📌 Save this reel and start planning your next adventure!\n\n"
            f"#SkysafarTourism #{DESTINATION.replace(' ', '')} #Travel #Wanderlust"
        )
        send_video_telegram(video_path, caption)
        print("\n✅ Done! Reel sent to Telegram.")
    finally:
        cleanup_temp_files(audio_path, video_path)

if __name__ == "__main__":
    main()

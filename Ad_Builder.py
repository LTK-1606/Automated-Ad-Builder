import os
import sys
import subprocess
import pandas as pd
import gdown
from flask import Flask, render_template, request, redirect, url_for, session
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from moviepy import VideoFileClip, concatenate_videoclips
import webbrowser
from threading import Timer

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

if hasattr(sys, '_MEIPASS'):
    EXTERNAL_DIR = os.path.dirname(sys.executable)
else:
    EXTERNAL_DIR = os.path.abspath(".")

app = Flask(__name__, template_folder=get_resource_path("templates"))

TEMP_DIR = os.path.join(EXTERNAL_DIR, "temp_clips")
OUTPUT_DIR = os.path.join(EXTERNAL_DIR, "Output")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_ad_output.mp4")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if hasattr(sys, '_MEIPASS'):
    CLIP_PATH = os.path.join(EXTERNAL_DIR, "Video Clip Selector.xlsx")
else:
    CLIP_PATH = os.path.join(os.path.abspath("."), "Video Clip Selector.xlsx")

def reveal_output(file_path):
    path = os.path.normpath(file_path)
    if os.name == 'nt':  # Windows
        subprocess.Popen(f'explorer /select,"{path}"')
    elif sys.platform == 'darwin':  # macOS
        subprocess.Popen(['open', '-R', path])
    else:  # Linux/Unix
        subprocess.Popen(['xdg-open', os.path.dirname(path)])

app = Flask(__name__)
app.secret_key = "super_secret_session_key" 

print("Loading data and model...")
df = pd.read_excel(CLIP_PATH, sheet_name=1)
df.columns = df.columns.str.strip()
df_clips = df[['Clip Name', 'Actions', 'Clip Link']].copy()
df_clips.rename(columns={'Clip Name': 'clip_id', 'Actions': 'keywords', 'Clip Link': 'gdrive_url'}, inplace=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
clip_embeddings = model.encode(df_clips["keywords"].tolist(), show_progress_bar=False)
print("Startup complete!")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("script", "")
        ad_script = [line.strip() for line in user_input.split("\n") if line.strip()]
        
        if not ad_script:
            return render_template("index.html", error="Please enter a script.")
        
        scenes_data = []
        for line_number, script_line in enumerate(ad_script, start=1):
            script_embedding = model.encode([script_line])
            similarities = cosine_similarity(script_embedding, clip_embeddings)[0]
            top_5_indices = similarities.argsort()[-5:][::-1]
            
            choices = []
            for i, idx in enumerate(top_5_indices):
                choices.append({
                    "clip_id": str(df_clips.iloc[idx]["clip_id"]),
                    "keywords": str(df_clips.iloc[idx]["keywords"]),
                    "gdrive_url": str(df_clips.iloc[idx]["gdrive_url"]),
                    "score": f"{similarities[idx]:.1%}"
                })
                
            scenes_data.append({
                "line_number": line_number,
                "script_line": script_line,
                "choices": choices
            })
        
        session["ad_script"] = ad_script
        return render_template("select_clips.html", scenes=scenes_data)
        
    return render_template("index.html")

@app.route("/render", methods=["POST"])
def render_sequence():
    ad_script = session.get("ad_script", [])
    if not ad_script:
        return redirect(url_for("index"))
        
    downloaded_clip_paths = []
    
    for line_number in range(1, len(ad_script) + 1):
        clip_id = request.form.get(f"scene_{line_number}_clip_id")
        
        match_row = df_clips[df_clips['clip_id'] == clip_id]
        if match_row.empty:
            return f"Error: Clip metadata missing for selection in Scene {line_number}", 400
            
        gdrive_url = match_row.iloc[0]['gdrive_url']
        local_filename = os.path.join(TEMP_DIR, f"{clip_id}.mp4")
        
        if not os.path.exists(local_filename):
            try:
                gdown.download(gdrive_url, local_filename, quiet=True)
            except Exception as e:
                return f"Failed to download clip '{clip_id}'. Check Google Drive sharing permissions. Details: {e}", 500
                
        downloaded_clip_paths.append(local_filename)
        
    try:
        video_clips = [VideoFileClip(path) for path in downloaded_clip_paths]
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video.write_videofile(
            FINAL_OUTPUT_PATH, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac"
        )

        final_video.close()
        for clip in video_clips:
            clip.close()

        reveal_output(FINAL_OUTPUT_PATH)
 
    except Exception as e:
        return f"MoviePy compilation failed: {e}", 500
        
    return render_template("result.html")

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001")

if __name__ == "__main__":
    Timer(1.5, open_browser).start()
    
    app.run(debug=False, port=5001, use_reloader=False)
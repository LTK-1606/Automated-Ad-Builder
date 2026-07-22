import os
import sys
import multiprocessing
import subprocess
import pandas as pd
import numpy as np
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
    # PyInstaller compiled bundle
    exe_dir = os.path.dirname(sys.executable)
    
    if sys.platform == 'darwin' and '.app/Contents/MacOS' in exe_dir:
        # macOS Bundle (application)
        EXTERNAL_DIR = os.path.abspath(os.path.join(exe_dir, '../../..'))
    else:
        # Windows (.exe folder)
        EXTERNAL_DIR = exe_dir
else:
    # standard uncompiled python script
    EXTERNAL_DIR = os.path.abspath(".")

app = Flask(__name__, template_folder=get_resource_path("templates"))
app.secret_key = "super_secret_session_key" 

TEMP_DIR = os.path.join(EXTERNAL_DIR, "temp_clips")
OUTPUT_DIR = os.path.join(EXTERNAL_DIR, "Output")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_ad_output.mp4")
CLIP_PATH = os.path.join(EXTERNAL_DIR, "Video Clip Selector.xlsx")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def reveal_output(file_path):
    path = os.path.normpath(file_path)
    if os.name == 'nt':  # Windows
        subprocess.Popen(f'explorer /select,"{path}"')
    elif sys.platform == 'darwin':  # macOS
        subprocess.Popen(['open', '-R', path])
    else:  # Linux/Unix
        subprocess.Popen(['xdg-open', os.path.dirname(path)])

df_clips = None
model = None
clip_embeddings = None

def init_resources():
    global df_clips, model, clip_embeddings
    if model is not None:
        return  

    print("Loading data and model...")
    df = pd.read_excel(CLIP_PATH, sheet_name="VideoTimeStamps")
    df.columns = df.columns.str.strip()
    df_clips = df[['Clip Name', 'Actions', 'Clip Link']].copy()
    df_clips.rename(columns={'Clip Name': 'clip_id', 'Actions': 'keywords', 'Clip Link': 'gdrive_url'}, inplace=True)
    df_clips['Country'] = df['Country'] if 'Country' in df.columns else "Any"
    df_clips['Subject'] = df['Subject'] if 'Subject' in df.columns else "Any"

    model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_embeddings = model.encode(df_clips["keywords"].tolist(), show_progress_bar=False)
    print("Startup complete!")

@app.route("/", methods=["GET", "POST"])
def index():
    init_resources() 
    countries = sorted(df_clips['Country'].dropna().astype(str).unique().tolist())
    subjects = sorted(df_clips['Subject'].dropna().astype(str).unique().tolist())
    
    if request.method == "POST":
        user_input = request.form.get("script", "")
        ad_script = [line.strip() for line in user_input.split("\n") if line.strip()]
        
        selected_country = request.form.get("country", "Any")
        selected_subject = request.form.get("subject", "Any")
        
        if not ad_script:
            return render_template("index.html", error="Please enter a script.", countries=countries, subjects=subjects)
        
        mask = pd.Series(True, index=df_clips.index)
        
        if selected_country and selected_country != "Any":
            mask &= (df_clips['Country'].astype(str).str.strip().str.lower() == selected_country.strip().lower())
        if selected_subject and selected_subject != "Any":
            mask &= (df_clips['Subject'].astype(str).str.strip().str.lower() == selected_subject.strip().lower())
            
        filtered_df = df_clips[mask]
        
        if filtered_df.empty:
            return render_template("index.html", error="Error: No clips match this exact Country and Subject combination.", countries=countries, subjects=subjects)
            
        subset_embeddings = clip_embeddings[mask.to_numpy()]
        
        scenes_data = []
        for line_number, script_line in enumerate(ad_script, start=1):
            script_embedding = model.encode([script_line])
            
            similarities = cosine_similarity(script_embedding, subset_embeddings)[0]
            
            top_k = min(5, len(filtered_df))
            top_subset_indices = similarities.argsort()[-top_k:][::-1]
            
            choices = []
            for idx in top_subset_indices:
                choices.append({
                    "clip_id": str(filtered_df.iloc[idx]["clip_id"]),
                    "keywords": str(filtered_df.iloc[idx]["keywords"]),
                    "gdrive_url": str(filtered_df.iloc[idx]["gdrive_url"]),
                    "score": f"{similarities[idx]:.1%}"
                })
                
            scenes_data.append({
                "line_number": line_number,
                "script_line": script_line,
                "choices": choices
            })
        
        session["ad_script"] = ad_script
        return render_template("select_clips.html", scenes=scenes_data)
        
    return render_template("index.html", error=None, countries=countries, subjects=subjects)

@app.route("/render", methods=["POST"])
def render_sequence():
    init_resources() 
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
                return f"Failed to download clip '{clip_id}'. Check Google Drive permissions. Details: {e}", 500
                
        downloaded_clip_paths.append(local_filename)
        
    try:
        video_clips = [VideoFileClip(path) for path in downloaded_clip_paths]
        final_video = concatenate_videoclips(video_clips, method="compose")
        temp_audio_path = os.path.join(OUTPUT_DIR, "temp_audio_build.m4a")

        final_video.write_videofile(
            FINAL_OUTPUT_PATH, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac",
            temp_audiofile=temp_audio_path,
            remove_temp=True
        )

        final_video.close()
        for clip in video_clips:
            clip.close()

        reveal_output(FINAL_OUTPUT_PATH)

    except Exception as e:
        return f"MoviePy compilation failed: {e}", 500
        
    return render_template("result.html")

@app.route("/shutdown", methods=["POST"])
def shutdown():
    def kill_server():
        import time
        time.sleep(0.5)  
        os._exit(0)      
        
    import threading
    threading.Thread(target=kill_server).start()
    
    return "Shutting down...", 200

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001")

def kill_process_on_port(port):
    import platform
    try:
        if platform.system() == "Windows":
            cmd = f'netstat -ano | findstr :{port}'
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            if output:
                for line in output.splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and f":{port}" in parts[1]:
                        pid = parts[-1]
                        print(f"Port {port} is busy. Killing Windows PID {pid}...")
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            cmd = f'lsof -t -i:{port}'
            pids = subprocess.check_output(cmd, shell=True).decode().strip().split()
            for pid in pids:
                if pid:
                    print(f"Port {port} is busy. Killing macOS PID {pid}...")
                    subprocess.run(f'kill -9 {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass
    except Exception as e:
        print(f"Error trying to clear port {port}: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    PORT = 5001
    kill_process_on_port(PORT)
    init_resources()

    if not os.environ.get("WERKZEUG_RUN_MAIN") and not os.environ.get("APP_BROWSER_OPENED"):
        os.environ["APP_BROWSER_OPENED"] = "true"
        Timer(1.5, open_browser).start()
    
    app.run(debug=False, port=PORT, use_reloader=False)
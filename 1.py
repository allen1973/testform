#@title 🚀 終極自動化：MD5 去重 + AI 內容辨識分類 (預設預覽模式)
#@markdown ---
#@markdown ### 1. 路徑設定
source_path = "/content/drive/MyDrive/Unsorted" #@param {type:"string"}
target_base = "/content/drive/MyDrive/AI_Sorted_Result" #@param {type:"string"}
#@markdown ---
#@markdown ### 2. AI 與去重參數
confidence_threshold = 0.4 #@param {type:"slider", min:0, max:1, step:0.05}
dry_run = True #@param {type:"boolean"}
#@markdown ---

import os
import hashlib
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
from google.colab import drive

# 載入必要的 AI 庫
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# --- 【架構師】核心邏輯區 ---

def setup_environment():
    """初始化環境"""
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    return MobileNetV2(weights='imagenet')

def get_md5(file_path):
    """計算檔案 MD5 雜湊值 (分塊讀取)"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def predict_category(model, img_path):
    """AI 內容辨識"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)
        _, label, prob = decode_predictions(preds, top=1)[0][0]
        
        if prob >= confidence_threshold:
            # 格式化標籤：小寫且底線替代空格
            return label.lower().replace(" ", "_")
        return "uncertain_content"
    except Exception:
        return "error_processing"

# --- 【除錯官】防護執行區 ---

def main():
    model = setup_environment()
    src_dir = Path(source_path)
    
    if not src_dir.exists():
        print(f"❌ 錯誤：找不到來源路徑 {source_path}")
        return

    # 取得所有圖片
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = [f for f in src_dir.rglob('*') if f.suffix.lower() in extensions]
    
    print(f"📂 掃描完成：共計 {len(all_files)} 張圖片")
    
    seen_md5s = {}
    move_queue = [] # 格式: (原始路徑, 目標路徑, 理由)

    print("🧠 開始 MD5 去重與 AI 分析 (這可能需要一點時間)...")
    
    for f_path in all_files:
        # 1. MD5 去重
        f_hash = get_md5(f_path)
        if f_hash and f_hash in seen_md5s:
            dest_dir = Path(target_base) / "system_duplicates"
            reason = "Duplicate (MD5)"
        else:
            seen_md5s[f_hash] = f_path
            # 2. AI 辨識
            category = predict_category(model, f_path)
            dest_dir = Path(target_base) / category
            reason = f"AI Classified: {category}"
        
        move_queue.append((f_path, dest_dir / f_path.name, reason))

    # --- 執行階段 ---
    print(f"\n--- {'預覽模式' if dry_run else '正式執行模式'} ---")
    
    success_count = 0
    for src, dst, reason in move_queue:
        if dry_run:
            print(f"[預覽] {src.name} -> {dst.relative_to(Path(target_base).parent)} ({reason})")
        else:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                # 處理檔名衝突：若目標已存在同名檔，加上時間戳
                if dst.exists():
                    dst = dst.with_name(f"{dst.stem}_{datetime.now().strftime('%H%M%S')}{dst.suffix}")
                shutil.move(str(src), str(dst))
                success_count += 1
            except Exception as e:
                print(f"⚠️ 移動失敗 {src.name}: {e}")

    if not dry_run:
        print(f"\n✅ 任務完成！成功搬移 {success_count} 個檔案。")
    else:
        print(f"\n💡 預覽結束。若滿意結果，請取消勾選 dry_run 後重新執行。")

if __name__ == "__main__":
    main()

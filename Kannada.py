import os
font_path = "NotoSansKannada-Regular.ttf"

if os.path.exists(font_path):
    print("✅ Font file found!")
else:
    print("❌ Font file NOT found! Place it in the correct directory.")

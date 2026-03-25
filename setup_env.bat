@echo off
setlocal
REM ---- Chạy tại thư mục của script ----
cd /d "%~dp0"

REM ---- Đường dẫn python (sửa nếu khác) ----
set "PY=C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"

echo [1/4] Tạo virtual env...
"%PY%" -m venv .venv || goto :err

echo [2/4] Kích hoạt env...
call .\.venv\Scripts\activate.bat || goto :err

echo [3/4] Nâng pip...
python -m pip install --upgrade pip || goto :err

echo [4/4] Cài thư viện...
REM Nếu bạn có requirements.txt thì dùng dòng dưới (bỏ comment):
REM pip install -r requirements.txt || goto :err

pip install streamlit numpy matplotlib wordcloud tensorflow praw ^
 google-api-python-client google-auth-oauthlib google-auth ^
 youtube-transcript-api yt-dlp python-dotenv || goto :err

echo ========= DONE =========
echo Da cai xong moi thu. An phim bat ky de dong cua so.
pause
exit /b 0

:err
echo *** CO LOI XAY RA. XEM DONG O TREN ***
pause
exit /b 1

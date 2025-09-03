# Tạo môi trường ảo
python -m venv envlocal

# đổi file /envlocal/pyvenv.cfg thành đúng với chỗ chứa và phiên bản

home = C:\Users\daoho\AppData\Local\Programs\Python\Python310
include-system-site-packages = false
version = 3.10.0
executable = C:\Users\daoho\AppData\Local\Programs\Python\Python310\python.exe
command = C:\Users\daoho\AppData\Local\Programs\Python\Python310\python.exe -m venv D:\AI_HealthCare\mainAI\envlocal

# Cài thư viện 

pip install -r requirements.txt

# Sau khi xong thì 

python run.py
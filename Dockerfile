# start from python base image
FROM python:3.10

# change working directory
WORKDIR /code

# add requirements file to image
COPY ./requirements.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# add python code
COPY ./main.py /code/main.py  

# specify default commands
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]


# Step 1: FastAPI
#  Step 2: Docker
# ไปสร้าง build docker image
# rename Dockerfile.txt Dockerfile
# docker build -t physics-api .  #สร้าง docker image
# docker run -p 8000:80 physics-api

#vStep 3: Docker Hub
# docker image ขึ้น Docker Hub 
# docker login
# docker tag physics-api ppearppss/physics-api
# docker push ppearppss/physics-api

# Step 4: AWS ECS
# เปิก ECS
# Cluster  → กล่องใหญ่ที่เอาไว้รวม container ทั้งหมด
#     ↓
# Task Definition → สูตร/blueprint ของ container (ใช้ image อะไร, RAM เท่าไหร่)
#     ↓
# Service → ตัวที่รัน container จริงๆ ตาม Task Definition
# 1. คลิ๊กคลัสเตอร์ สร้างซะ
# 2. สร้าง Repository name
# 3. push image จากเครื่องคุณขึ้น ECR  
# git clone <repo ของคุณ>
# docker build -t physics-api .
# aws ecr get-login-password ...
# docker tag ...
# docker push ...
# 2. สร้าง Task Definition 
# สร้าง Service เพื่อให้มันรันจริง + ได้ Public URL

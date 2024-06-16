# Install.md
## 환경
OS : Ubuntu 23.10 <br>
GPU : NVIDIA GeForce RTX 4080 Laptop GPU <br> 
Cuda Version : 12.1 

<br><br>

## 설치
1. 가상환경 생성
    ```sh
    conda create -n <env_name> python=3.8
    conda activate  <env_name>
    ```
<br>

2. Pytorch 설치
- Cuda version에 맞게 설치

<br>

3. 필요한 package 설치
    ```sh
    pip install -r requirements.txt
    ```

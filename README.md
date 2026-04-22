# AutoScout

Setup and run guide.

## 1) Create Conda Environment

From project root:

```bash
cd "AutoScout"
conda env create -n AutoScout_Env --file environment.yml
conda activate AutoScout_Env
pip install -r requirements.txt
```


## 2) Start Backend

```bash
cd "AutoScout/backend/code"
python3 api_server.py
```

## 3) Start Frontend

Open a second terminal and run:

```bash
cd "AutoScout/frontend"
python3 -m http.server 5500
```

## 4) Open in Browser

Go to:

```text
http://localhost:5500
```

## 5) Demo Video Path

For a quick demo run, use this project-relative path in the **Video Path** box:

```text
../Test_Data/test_video1.mov
```

This resolves to:

```text
AutoScout/Test_Data/test_video1.mov
```


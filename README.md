# UAS MPML - OnlineFoods Classification

## Run locally
```bash
conda create -n mpml python=3.10 -y
conda activate mpml
pip install -r requirements.txt
python train_onlinefoods.py
```

## Deploy to Heroku
1. Commit all files to Git
2. `heroku create yourappname`
3. `git push heroku main`

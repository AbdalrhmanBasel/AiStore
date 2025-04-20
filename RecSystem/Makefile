install:
    pip install -r requirements.txt

preprocess:
    python data_preprocessing/preprocess.py

train:
    python model_training/train.py

evaluate:
    python model_training/evaluate.py

deploy:
    python deployment/deploy_model.py

monitor:
    python monitoring/monitor.py

test:
    pytest tests/

clean:
    rm -rf __pycache__
    rm -rf *.pyc

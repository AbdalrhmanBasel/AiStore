install:
	pip install -e recommendation

train:
	python recommendation/scripts/train_model.py

preprocess:
	python recommendation/scripts/preprocess_data.py

test:
	pytest recommendation/tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

PYTHON = python3
PIP = pip
VENV = .env
REQUIREMENTS = requirements.txt
DATA_DIR = data
MODEL_DIR = model
OUTPUT_DIR = output
LOG_DIR = logs


install-requirements:
	@echo "Installing requirements..."
	pip install -r $(REQUIREMENTS)


preprocess: install-requirements
	@echo "Preprocessing data..."
	pip preprocessing.py --data-dir $(DATA_DIR)


train-model: preprocess
	@echo "Training model..."
	pip train.py --model-dir $(MODEL_DIR) --log-dir $(LOG_DIR)


test-model: train-model
	@echo "Testing model..."
	pip test.py --model-dir $(MODEL_DIR) --log-dir $(LOG_DIR)


evaluate: test-model
	@echo "Evaluating model..."
	pip evaluate.py --model-dir $(MODEL_DIR) --log-dir $(LOG_DIR) --output-dir $(OUTPUT_DIR)


test: preprocess train-model test-model evaluate
	@echo "All steps completed."


clean:
	@echo "Cleaning up..."
	rm -rf $(VENV) $(DATA_DIR)/*.pt $(MODEL_DIR)/*.h5 $(OUTPUT_DIR)/* $(LOG_DIR)/*
	@echo "Clean up complete."

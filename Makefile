ENV_FOLDER=./environment
VENV_NAME=___venv
VENV_PATH=$(ENV_FOLDER)/$(VENV_NAME)
VENV_ACTIVATE_PATH=$(VENV_PATH)/bin/activate
REQUIREMENTS_PATH=$(ENV_FOLDER)/requirements.txt
PYTHON_VERSION = python3.8


create-env:
	@echo "======================== Creating the project virtual environment ========================" 
	$(PYTHON_VERSION) -m virtualenv --system-site-packages -p $(PYTHON_VERSION) $(VENV_PATH)
	. $(VENV_ACTIVATE_PATH) && \
	$(PYTHON_VERSION) -m pip install pip --upgrade && \
	$(PYTHON_VERSION) -m pip install --upgrade six && \
	$(PYTHON_VERSION) -m pip install -r $(REQUIREMENTS_PATH)

activate-env-command:
	@echo "======================== Execute the below command in terminal ========================" 
	@echo source $(VENV_ACTIVATE_PATH)

download-dataset:
	. $(VENV_ACTIVATE_PATH) && \
	cd data && \
	kaggle datasets download mathurinache/star-wars-images 	&& \
	mkdir star-wars-images &&\
	unzip star-wars-images.zip -d star-wars-images && \
	rm star-wars-images.zip 

run-training:
	. $(VENV_ACTIVATE_PATH) && \
	cd src/ && \
	python train.py

purge-output:
	rm -r output/lightning_logs/version_*

run-tensorboard:
	@echo "======================== Run the displayed link in your browser to view training results via tensorboard ========================" 
	. $(VENV_ACTIVATE_PATH) && \
	tensorboard --logdir ./output/

SHELL=/bin/bash

install-external-pkgs:
	@echo "Installing external dependencies..."
	source env_vars.env &&\
		./scripts/utils/install-pkgs.sh
		
download-data:
	@echo "Downloading data..."
	source env_vars.env && \
		./scripts/data_acquisition/download-data.sh
		
organize-data:
	@echo "Uncompressing data..."
	source env_vars.env &&\
		./scripts/data_acquisition/organize-data.sh

handle-signal-data:
	@echo "Handling signal features data"
	source env_vars.env &&\
		python scripts/utils/handle_signal_feat_data.py --path ${COMPLETE_DATA_PATH}

run-pipe-signal-fts-3-secs-svm:
	@echo "Executing pipeline for signal feat data"
	source env_vars.env &&\
		python scripts/training/training_signal_fts.py --track_sec_length 3

run-pipe-signal-fts-30-secs-svm:
	@echo "Executing pipeline for signal feat data"
	source env_vars.env &&\
		python scripts/training/training_signal_fts.py --track_sec_length 30

run-pipe-signal-fts-3-secs-rand_for:
	@echo "Executing pipeline for signal feat data"
	source env_vars.env &&\
		python scripts/training/training_signal_fts_forest.py --track_sec_length 3

run-pipe-signal-fts-30-secs-ran_for:
	@echo "Executing pipeline for signal feat data"
	source env_vars.env &&\
		python scripts/training/training_signal_fts_forest.py --track_sec_length 30
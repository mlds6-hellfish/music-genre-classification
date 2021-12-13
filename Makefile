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

###############################################################################
# HELP / DEFAULT COMMAND
###############################################################################
.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)


.PHONY: install_tk
install_tk: ## Install tkinter used by matplotlib backend
	pip install python3.6-tk


.PHONY: freeze
freeze: ## Freeze requirements
	pip freeze > requirements.txt
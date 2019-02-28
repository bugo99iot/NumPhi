###############################################################################
# HELP / DEFAULT COMMAND
###############################################################################
.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: freeze
freeze: ## Freeze requirements
	pip freeze > requirements.txt

.PHONY: test
test: ## run pytest on local machine
	. venv/bin/activate && cd numphi && pytest -vv


.PHONY: pip-deploy
pip-deploy: ## Deploy package to pip universe
	make test
	# git commit here
	bumpversion patch
	python setup.py sdist
	# sudo rm -R dist


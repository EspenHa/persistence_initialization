.PHONY: pull build deploy

pull:
	docker pull espenha/research:persistence_initialization

build:
	docker build -t espenha/research:persistence_initialization . 

deploy:
	git add -A
	-git rm $$(git ls-files --deleted) 2> /dev/null
	git commit --allow-empty --no-verify --no-gpg-sign -m "TEMPORARY DEPLOY COMMIT"
	-git push --force --no-verify ${DEPLOY_TARGET} HEAD:deploy
	git reset HEAD~

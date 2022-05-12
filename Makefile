distr:
	python3 setup.py bdist_wheel

push:
	twine upload dist/*

test:
	pytest
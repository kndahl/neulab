distr:
	rm -rf build
	mkdir build
	rm -rf dist
	mkdir dist
	python3 setup.py bdist_wheel

push:
	twine upload dist/*

test:
	pytest
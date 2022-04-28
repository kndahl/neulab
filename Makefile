distr:
	python3 setup.py sdist

distrwheel:
	python3 setup.py bdist_wheel

test:
	pytest
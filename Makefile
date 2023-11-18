
default: pytest

pytest:
	echo "no tests"

install_requirements:
	pip install -r requirements.txt

install_requirements_dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

streamlit:
	streamlit run app.py

install:
	pip install . -U

clean:
	rm -fr */__pycache__
	rm -fr __init__.py
	rm -fr build
	rm -fr dist
	rm -fr *.dist-info
	rm -fr *.egg-info

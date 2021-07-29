### Instruction for launching tests of Model Analyzer on Ubuntu

> NOTE: It is recommended to use Python virtual environment to work with the repository

1. [Install OpenVINO package](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html) 

2. **OPTIONAL** Create and activate Python virtual environment:
```shell
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
```

3. Install required packages:
```shell
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

3. Initialize the OpenVINO environment:
```shell
source ~/intel/openvino_2021/bin/setupvars.sh
```

4. Run the script to download models for tests:
```shell
python tests/download_models.py --config tests/data/IRv10_models.json
python tests/download_models.py --config tests/data/onnx_models.json
```

5. Set environment variables to the downloaded models directory:
```shell
export MODELS_PATH=tests/data/models
```
6. Run the tests
```shell
pytest --disable-warnings -r A
```

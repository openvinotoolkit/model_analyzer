### Instruction for launching tests of Model Analyzer on Ubuntu

> NOTE: The Model Analyzer branches are synced with branches in the [OpenVINO](https://github.com/openvinotoolkit/openvino) repository.

It is recommended to use Python virtual environment to work with the repository.
1. **OPTIONAL** Create and activate Python virtual environment:
```shell
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
```

2. Install required packages:
```shell
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

3. Run the script to download models for tests:
```shell
python tests/download_models.py --config tests/data/IRv10_models.json
python tests/download_models.py --config tests/data/onnx_models.json
```

4. Set environment variables to the downloaded models directory:
```shell
export MODELS_PATH=tests/data/models
```
5. Run the tests:
```shell
pytest --disable-warnings -r A
```

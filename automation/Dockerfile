FROM ${OPENVINO_IMAGE_REPOSITORY}:2022.1.0.347_tgl

ENV USERNAME openvino

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV OPENVINO_MODEL_ANALYZER_ROOT /home/${USERNAME}/model_analyzer

RUN mkdir ${OPENVINO_MODEL_ANALYZER_ROOT}

COPY --chown=openvino requirements.txt ${OPENVINO_MODEL_ANALYZER_ROOT}/
COPY --chown=openvino requirements_dev.txt ${OPENVINO_MODEL_ANALYZER_ROOT}/

WORKDIR ${OPENVINO_MODEL_ANALYZER_ROOT}

RUN python3 -m pip install --no-cache-dir --proxy=${https_proxy} -r ./requirements.txt && \
    python3 -m pip install --no-cache-dir --proxy=${https_proxy} -r ./requirements_dev.txt

COPY --chown=openvino . ${OPENVINO_MODEL_ANALYZER_ROOT}

USER ${USERNAME}

WORKDIR ${OPENVINO_MODEL_ANALYZER_ROOT}/

ENTRYPOINT []

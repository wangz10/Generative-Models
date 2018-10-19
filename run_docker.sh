#!/bin/bash
docker run -it -p 8888:8888 \
    -v $(pwd):/home/jovyan \
    jupyter/tensorflow-notebook:5811dcb711ba
    
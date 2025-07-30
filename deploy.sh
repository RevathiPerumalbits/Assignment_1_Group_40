#!/bin/bash
docker pull revathiperumalbits/iris-mlops-api:latest
docker run -d -p 8000:8000 revathiperumalbits/iris-mlops-api:latest
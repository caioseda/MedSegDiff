#!/bin/bash

echo "Digite seu GitHub Token (ele nao sera exibido):"
read -s GITHUB_TOKEN

docker build \
  --build-arg GITHUB_TOKEN=$GITHUB_TOKEN \
  -t medsegdiff-vfss:latest .
#!/bin/bash

# echo "Digite seu GitHub Token (ele nao sera exibido):"
# read -s GITHUB_TOKEN

echo "Construindo a imagem Docker medsegdiff-vfss:latest..."
echo "Removendo pasta vfss-data-split..."
rm -rf ./vfss-data-split
echo "./vfss-data-split removido com sucesso."

echo "Copiando pasta vfss-data-split..."
cp -r ../vfss-data-split/ ./
echo "Pasta vfss-data-split copiada com sucesso."

echo "Iniciando a construcao da imagem Docker..."
docker build \
  --build-arg GITHUB_TOKEN=$GITHUB_TOKEN \
  -t medsegdiff-vfss:latest .
echo "Imagem Docker medsegdiff-vfss:latest construída com sucesso."
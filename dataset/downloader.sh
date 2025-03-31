#!/usr/bin/env sh
echo "Download music playlist. Source: https://www.cs.cornell.edu/~shuochen/lme/data_page.html"

if [ ! -d $(pwd)/music_playlist ]; then
  wget -nc https://www.cs.cornell.edu/~shuochen/lme/dataset.tar.gz
  tar -xf dataset.tar.gz && mv dataset music_playlist
else
  echo "SKIP: dataset already exists"
fi
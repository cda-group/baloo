#!/bin/bash

cp "$TRAVIS_BUILD_DIR/weld/target/release/libweld.dylib" "$TRAVIS_BUILD_DIR/baloo/weld/libs/libweld.dylib"
make -C "$TRAVIS_BUILD_DIR/baloo/weld/convertors"
pipenv install --dev

# make sure we return here
cd $TRAVIS_BUILD_DIR

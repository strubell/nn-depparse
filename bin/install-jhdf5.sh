#!/bin/bash

LIB_DIR="$NNDEPPARSE_ROOT/lib"
LIB_VERSION="14.2.5-r35810"
LIB_URL="https://wiki-bsse.ethz.ch/download/attachments/26609237/sis-jhdf5-14.12.5-r35810.zip?version=1&modificationDate=1457264141573&api=v2"
LIB_ZIP="sis-jhdf5.zip"
IVY_FILE="$HOME/.ivy2/local/sis/jhdf5_2.11/14.2.5-r35810/jars/jhdf5_2.11.jar"

# make directory to contain lib if it doesn't exist
if [ ! -d $LIB_DIR ]; then
  mkdir -p $LIB_DIR
fi

printf "Checking for jhdf5...\n"
if [ ! -f $IVY_FILE ]; then
  printf "Installing jhdf5 into $LIB_DIR...\n"
  START_DIR=`pwd`
  cd $LIB_DIR
  wget "${LIB_URL}" -O ${LIB_ZIP}
  unzip ${LIB_ZIP}
  cp sis-jhdf5/lib/batteries_included/sis-jhdf5-batteries_included.jar .

  printf "organization := \"sis\"\n\n" > build.sbt
  printf "name := \"jhdf5\"\n\n" >> build.sbt
  printf "version := \"$LIB_VERSION\"\n\n" >> build.sbt
  printf "scalaVersion := \"2.11.7\"\n\n" >> build.sbt
  printf "packageBin in Compile := file(s\"\${organization.value}-\${name.value}-batteries_included.jar\")\n" >> build.sbt

  if [ ! -d project ]; then
    mkdir project
  fi
  echo "sbt.version=0.13.9" > project/build.properties
  sbt publishLocal
  rm -r sis-jhdf5
  rm ${LIB_ZIP}
  cd $START_DIR
fi


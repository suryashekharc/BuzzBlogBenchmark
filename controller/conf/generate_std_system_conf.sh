#!/bin/bash

# Copyright (C) 2022 Georgia Tech Center for Experimental Research in Computer
# Systems

# Process command-line arguments.
set -u
while [[ $# > 1 ]]; do
  case $1 in
    --hardware )
      hardware=$2
      ;;
    * )
      echo "Invalid argument: $1"
      exit 1
  esac
  shift
  shift
done

# Set current directory to the directory of this script.
cd "$(dirname "$0")"

# Generate standard system configuration file.
./generate_system_confs.sh \
	--hardware ${hardware} \
	--cpu 4 \
	--gb 16 \
	--monitoringinterval 0.05 \
	--gunicornacceptqueue 4096 \
	--gunicornworkers 16 \
	--gunicornthreads 16 \
	--gunicornserviceconnpoolminsize 2 \
  --gunicornserviceconnpoolmaxsize 2 \
  --gunicornserviceconnpoolalloweph 1 \
	--thriftacceptqueue 4096 \
	--thriftthreads 512 \
	--thriftserviceconnpoolminsize 32 \
  --thriftserviceconnpoolmaxsize 32 \
  --thriftserviceconnpoolalloweph 1 \
	--thriftpgconnpoolminsize 32 \
  --thriftpgconnpoolmaxsize 32 \
  --thriftpgconnpoolalloweph 0 \
	--thriftredisconnpoolsize 32 \
	--pgmaxconnections 32 \
	--redisacceptqueue 4096 \
	--redismaxclients 32 \
	--redissnapshotinterval 60 \
	--ninvalidwords 4096

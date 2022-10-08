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
    --cpu )
      cpu=$2
      ;;
    --gb )
      gb=$2
      ;;
    --monitoringinterval )
      monitoringinterval=$2
      ;;
    --gunicornacceptqueue )
      gunicornacceptqueue=$2
      ;;
    --gunicornworkers )
      gunicornworkers=$2
      ;;
    --gunicornthreads )
      gunicornthreads=$2
      ;;
    --gunicornserviceconnpoolminsize )
      gunicornserviceconnpoolminsize=$2
      ;;
    --gunicornserviceconnpoolmaxsize )
      gunicornserviceconnpoolmaxsize=$2
      ;;
    --gunicornserviceconnpoolalloweph )
      gunicornserviceconnpoolalloweph=$2
      ;;
    --thriftacceptqueue )
      thriftacceptqueue=$2
      ;;
    --thriftthreads )
      thriftthreads=$2
      ;;
    --thriftserviceconnpoolminsize )
      thriftserviceconnpoolminsize=$2
      ;;
    --thriftserviceconnpoolmaxsize )
      thriftserviceconnpoolmaxsize=$2
      ;;
    --thriftserviceconnpoolalloweph )
      thriftserviceconnpoolalloweph=$2
      ;;
    --thriftpgconnpoolminsize )
      thriftpgconnpoolminsize=$2
      ;;
    --thriftpgconnpoolmaxsize )
      thriftpgconnpoolmaxsize=$2
      ;;
    --thriftpgconnpoolalloweph )
      thriftpgconnpoolalloweph=$2
      ;;
    --thriftredisconnpoolsize )
      thriftredisconnpoolsize=$2
      ;;
    --pgmaxconnections )
      pgmaxconnections=$2
      ;;
    --redisacceptqueue )
      redisacceptqueue=$2
      ;;
    --redismaxclients )
      redismaxclients=$2
      ;;
    --redissnapshotinterval )
      redissnapshotinterval=$2
      ;;
    --ninvalidwords )
      ninvalidwords=$2
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

# Generate system configuration files.
for it_cpu in $cpu; do
  for it_gb in $gb; do
    for it_monitoringinterval in $monitoringinterval; do
      for it_gunicornacceptqueue in $gunicornacceptqueue; do
        for it_gunicornworkers in $gunicornworkers; do
          for it_gunicornthreads in $gunicornthreads; do
            for it_gunicornserviceconnpoolminsize in $gunicornserviceconnpoolminsize; do
              for it_gunicornserviceconnpoolmaxsize in $gunicornserviceconnpoolmaxsize; do
                for it_gunicornserviceconnpoolalloweph in $gunicornserviceconnpoolalloweph; do
                  for it_thriftacceptqueue in $thriftacceptqueue; do
                    for it_thriftthreads in $thriftthreads; do
                      for it_thriftserviceconnpoolminsize in $thriftserviceconnpoolminsize; do
                        for it_thriftserviceconnpoolmaxsize in $thriftserviceconnpoolmaxsize; do
                          for it_thriftserviceconnpoolalloweph in $thriftserviceconnpoolalloweph; do
                            for it_thriftpgconnpoolminsize in $thriftpgconnpoolminsize; do
                              for it_thriftpgconnpoolmaxsize in $thriftpgconnpoolmaxsize; do
                                for it_thriftpgconnpoolalloweph in $thriftpgconnpoolalloweph; do
                                  for it_thriftredisconnpoolsize in $thriftredisconnpoolsize; do
                                    for it_pgmaxconnections in $pgmaxconnections; do
                                      for it_redisacceptqueue in $redisacceptqueue; do
                                        for it_redismaxclients in $redismaxclients; do
                                          for it_redissnapshotinterval in $redissnapshotinterval; do
                                            for it_ninvalidwords in $ninvalidwords; do
                                              filename="BuzzBlog-19_"
                                              filename+="${it_cpu}CPU_"
                                              filename+="${it_gb}GB_"
                                              filename+="${it_gunicornacceptqueue}GACCEPTQ_"
                                              filename+="${it_gunicornworkers}GWORKERS_"
                                              filename+="${it_gunicornthreads}GTHREADS_"
                                              filename+="${it_gunicornserviceconnpoolminsize}GMINCP_"
                                              filename+="${it_gunicornserviceconnpoolmaxsize}GMAXCP_"
                                              filename+="${it_thriftacceptqueue}TACCEPTQ_"
                                              filename+="${it_thriftthreads}TTHREADS_"
                                              filename+="${it_thriftserviceconnpoolminsize}TSMINCP_"
                                              filename+="${it_thriftserviceconnpoolmaxsize}TSMAXCP_"
                                              filename+="${it_thriftpgconnpoolminsize}TPGMINCP_"
                                              filename+="${it_thriftpgconnpoolmaxsize}TPGMAXCP_"
                                              filename+="${it_thriftredisconnpoolsize}TREDISCP_"
                                              filename+="${it_pgmaxconnections}PGMAXCONN_"
                                              filename+="${it_redisacceptqueue}RACCEPTQ_"
                                              filename+="${it_redismaxclients}RMAXCLIENTS.yml"
                                              it_monitoringintervalinms=$(echo "(${it_monitoringinterval} * 1000) / 1" | bc)
                                              cp BuzzBlog-19_${hardware}_${it_cpu}CPU_TEMPLATE.yml $filename
                                              sed -i "s/{{GB}}/${it_gb}g/g" $filename
                                              sed -i "s/{{MONITORINGINTERVAL}}/${it_monitoringinterval}/g" $filename
                                              sed -i "s/{{MONITORINGINTERVALINMS}}/${it_monitoringintervalinms}/g" $filename
                                              sed -i "s/{{GUNICORNACCEPTQUEUE}}/${it_gunicornacceptqueue}/g" $filename
                                              sed -i "s/{{GUNICORNWORKERS}}/${it_gunicornworkers}/g" $filename
                                              sed -i "s/{{GUNICORNTHREADS}}/${it_gunicornthreads}/g" $filename
                                              sed -i "s/{{GUNICORNSERVICECONNPOOLMINSIZE}}/${it_gunicornserviceconnpoolminsize}/g" $filename
                                              sed -i "s/{{GUNICORNSERVICECONNPOOLMAXSIZE}}/${it_gunicornserviceconnpoolmaxsize}/g" $filename
                                              sed -i "s/{{GUNICORNSERVICECONNPOOLALLOWEPH}}/${it_gunicornserviceconnpoolalloweph}/g" $filename
                                              sed -i "s/{{THRIFTACCEPTQUEUE}}/${it_thriftacceptqueue}/g" $filename
                                              sed -i "s/{{THRIFTTHREADS}}/${it_thriftthreads}/g" $filename
                                              sed -i "s/{{THRIFTSERVICECONNPOOLMINSIZE}}/${it_thriftserviceconnpoolminsize}/g" $filename
                                              sed -i "s/{{THRIFTSERVICECONNPOOLMAXSIZE}}/${it_thriftserviceconnpoolmaxsize}/g" $filename
                                              sed -i "s/{{THRIFTSERVICECONNPOOLALLOWEPH}}/${it_thriftserviceconnpoolalloweph}/g" $filename
                                              sed -i "s/{{THRIFTPGCONNPOOLMINSIZE}}/${it_thriftpgconnpoolminsize}/g" $filename
                                              sed -i "s/{{THRIFTPGCONNPOOLMAXSIZE}}/${it_thriftpgconnpoolmaxsize}/g" $filename
                                              sed -i "s/{{THRIFTPGCONNPOOLALLOWEPH}}/${it_thriftpgconnpoolalloweph}/g" $filename
                                              sed -i "s/{{THRIFTREDISCONNPOOLSIZE}}/${it_thriftredisconnpoolsize}/g" $filename
                                              sed -i "s/{{PGMAXCONNECTIONS}}/${it_pgmaxconnections}/g" $filename
                                              sed -i "s/{{REDISACCEPTQUEUE}}/${it_redisacceptqueue}/g" $filename
                                              sed -i "s/{{REDISMAXCLIENTS}}/${it_redismaxclients}/g" $filename
                                              sed -i "s/{{REDISSNAPSHOTINTERVAL}}/${it_redissnapshotinterval}/g" $filename
                                              sed -i "s/{{NINVALIDWORDS}}/${it_ninvalidwords}/g" $filename
                                            done
                                          done
                                        done
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

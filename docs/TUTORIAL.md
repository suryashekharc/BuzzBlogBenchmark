# Tutorial
This is a tutorial for Georgia Tech students enrolled in *Introduction to
Enterprise Computing* (CS4365/6365) or *Real-Time/Embedded Systems*
(CS4220/CS6235). It shows how to run a *BuzzBlog Benchmark* experiment on a
cloud instantiated in CloudLab.

If you have not created and configured your CloudLab account, read the
[tutorial on how to setup a CloudLab account](CLOUDLAB.md) first.

## Cloud Instantiation
1. Access the [CloudLab login page](https://cloudlab.us/login.php) and sign in.
2. In the main menu, click on
[Start Experiment](https://www.cloudlab.us/instantiate.php).
3. Click on *Change Profile*.
4. Search for profile *BuzzBlog-19_xl170_nodes_STD* of project *Infosphere*.
This profile contains a cloud specification: 19 xl170 machines (x86
architecture) running Ubuntu 20.04 LTS (focal) 64-bit. Click on *Select
Profile*.
5. Optionally, you can give a name to your experiment. Click on *Next*.
6. Optionally, you can set a start time and duration for your experiment. The
default duration is 16 hours, but you will need much less time (3 hours should
be enough).
7. Click on *Finish*.

Your cloud will be ready in approximately 10 minutes, if the nodes requested are
available.

## Controller Setup
The experiment execution workflow is orchestrated by `benchmarkcontroller`, a
containerized application that installs software dependencies, starts services
and monitors, runs workload generators, and collects log files.

To setup `node-0` for running `benchmarkcontroller`:
1. Download `scripts/controller_setup.sh` in your local machine.
2. Run the script:
```
./controller_setup.sh \
    --username [your cloudlab username] \
    --private_ssh_key_path [path to your private ssh key] \
    --controller_node [node-0 hostname]
```

This SSH private key must be the one associated with the SSH public key uploaded
to CloudLab.

## SSH Server Setup
`benchmarkcontroller` is a multithreaded application whose threads share SSH
connections to run commands in remote nodes in parallel.

To increase the maximum number of sessions allowed per SSH connection in every
node:
1. Download `scripts/sshd_setup.sh` in your local machine.
2. Run the script:
```
./sshd_setup.sh \
    --username [your cloudlab username] \
    --controller_node [node-0 hostname] \
    --n_nodes 19
```

## Log Storage Setup
The size of system and application monitoring log files can be in the order of
gigabytes for experiments with high workloads.

To setup and mount large disk partitions on `/tmp`, where these log files are
stored:
1. Download `scripts/tmp_directory_setup.sh` in your local machine.
2. Run the script:
```
./tmp_directory_setup.sh \
    --username [your cloudlab username] \
    --controller_node [node-0 hostname] \
    --n_nodes 19 \
    --partition /dev/sda4
```

## Experiment Configuration
To generate experiment configuration files in `node-0`:
1. Download `scripts/tutorial_configuration_setup.sh` in your local machine.
2. Run the script:
```
./tutorial_configuration_setup.sh \
    --username [your cloudlab username] \
    --controller_node [node-0 hostname] \
    --system_template BuzzBlog-19_xl170_4CPU.yml \
    --workload_template BuzzBlog-bursty_workload.yml
```

Log into `node-0`:
```
ssh [your cloudlab username]@[node-0 hostname]
```

In the home folder, you will find experiment configuration files `system.yml`
and `workload.yml`.

`system.yml` contains the system configuration of each node: kernel parameters
to be overwritten, containers to be deployed and their configurations, system
monitors to be used and their options, etc. For a better understanding of your
experiment, read this file and how it is used by `benchmarkcontroller`
(specifically, look at Python script `controller/src/run_experiment.py`).

`workload.yml` contains the workload configuration: number of clients, mean
think time between consecutive requests, burstiness parameters, and
probabilities of transitioning between request types. For a better understanding
of your experiment, read this file and how it is used by `loadgen` for
generating requests that simulate user interactions with BuzzBlog (specifically,
look at Python script `loadgen/loadgen.py`).

## Experiment Execution
In `node-0`, start `benchmarkcontroller`:
```
sudo docker run \
    --name benchmarkcontroller \
    --env description="My first BuzzBlog experiment." \
    --env docker_hub_username="" \
    --env docker_hub_password="" \
    --volume $(pwd):/usr/local/etc/BuzzBlogBenchmark \
    --volume /tmp:/var/log/BuzzBlogBenchmark \
    --volume $(pwd)/.ssh:/home/$(whoami)/.ssh \
    $(echo $(cat /etc/hosts | grep node- | sed 's/[[:space:]]/ /g' | cut -d ' ' -f 1,4 | sed 's:^\(.*\) \(.*\):\2\:\1:' | awk '{print "--add-host="$1""}')) \
    rodrigoalveslima/buzzblog:benchmarkcontroller_v1.0
```

If you want to run a sequence of experiments:
1. Add system configuration files to a directory named `system` in the home
folder.
2. Add workload configuration files to a directory named `workload` in the home
folder.
3. Use the following options when starting `benchmarkcontroller`:
`--env workload_conf="/usr/local/etc/BuzzBlogBenchmark/workload"` and
`--env system_conf="/usr/local/etc/BuzzBlogBenchmark/system"`.

The experiment execution will take approximately 15 minutes, and its results
will be in a directory named `BuzzBlogBenchmark_[%Y-%m-%d-%H-%M-%S]` in the
`/tmp` directory.

After the experiment execution is finished, compress that directory:
```
cd /tmp
tar -czf $(ls . | grep BuzzBlogBenchmark_).tar.gz BuzzBlogBenchmark_*/*
```

Then, save the experiment results in your local machine:
```
scp [your cloudlab username]@[node-0 hostname]:/tmp/BuzzBlogBenchmark_*.tar.gz .
```

Finally, terminate the CloudLab experiment.

## Data Analysis
In directory `analysis`, you can find scripts and Jupyter notebooks to assist in
the analysis of experiment results.

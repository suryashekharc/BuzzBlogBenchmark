# Copyright (C) 2022 Georgia Tech Center for Experimental Research in Computer Systems
"""Plot system performance graphs for experiments."""

import argparse
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
from analysis.utils.utils import *


class LogAnalysis:

  def __init__(self, experiment_dirpath, output_dirpath):
    self._experiment_dirpath = experiment_dirpath
    self._output_dirpath = output_dirpath
    # Extract experiment information
    self._total_duration = get_experiment_total_duration(experiment_dirpath)
    self._ramp_up_duration = get_experiment_ramp_up_duration(experiment_dirpath)
    self._ramp_down_duration = get_experiment_ramp_down_duration(experiment_dirpath)
    self._node_names = get_node_names(experiment_dirpath)
    self._node_labels = {node_name: get_node_label(experiment_dirpath, node_name) for node_name in self._node_names}

  def plot(self, distribution):
    for attr_name in dir(self):
      if attr_name.startswith("plot_") and (distribution or "distribution" not in attr_name):
        getattr(self, attr_name)()

  def save_fig(func):

    def inner(self, *args, **kwargs):
      fig = func(self, *args, **kwargs)
      if self._output_dirpath and fig:
        fig.savefig(os.path.join(self._output_dirpath, "%s.png" % func.__name__))

    return inner


class RequestLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._requests = build_requests_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_number_of_successful_failed_requests(self):
    # Data frame
    df = self._requests[(self._requests.index >= self._ramp_up_duration) & (self._requests.index <= self._total_duration - self._ramp_down_duration)].\
        groupby(["status"]).count()["method"]
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(18, 6))
    ax = fig.gca()
    df.plot(ax=ax, kind="pie", title="Number of Successful/Failed Requests", xlabel="", ylabel="", legend=True)
    return fig

  @LogAnalysis.save_fig
  def plot_http_status_code_of_failed_requests(self):
    # Data frame
    df = self._requests[(self._requests["status"] == "failed") & (self._requests.index >= self._ramp_up_duration) &
                        (self._requests.index <= self._total_duration - self._ramp_down_duration)].groupby(["status_code"]).count()["method"]
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(18, 6))
    ax = fig.gca()
    df.plot(ax=ax, kind="pie", title="HTTP Status Code of Failed Requests", xlabel="", ylabel="", legend=True)
    return fig

  @LogAnalysis.save_fig
  def plot_number_of_read_write_requests(self):
    # Data frame
    df = self._requests[(self._requests.index >= self._ramp_up_duration) & (self._requests.index <= self._total_duration - self._ramp_down_duration)].\
        groupby(["rw"]).count()["method"]
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(18, 6))
    ax = fig.gca()
    df.plot(ax=ax, kind="pie", title="Number of Read/Write Requests", xlabel="", ylabel="", legend=True)
    return fig

  @LogAnalysis.save_fig
  def plot_number_of_requests_of_each_type(self):
    # Data frame
    df = self._requests[(self._requests.index >= self._ramp_up_duration) & (self._requests.index <= self._total_duration - self._ramp_down_duration)].\
        groupby(["type", "status"]).count()["method"].unstack().fillna(0)
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(18, 12))
    ax = fig.gca()
    ax.grid(alpha=0.75)
    df.plot(ax=ax,
            kind="bar",
            stacked=True,
            title="Number of Requests of Each Type",
            xlabel="",
            ylabel="Count (Requests)",
            color={
                "failed": "red",
                "successful": "blue"
            },
            legend=True,
            grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_latency_distribution_of_requests(self, latency_bin_in_ms=25, text_y=None):
    # Data frame
    df = self._requests[(self._requests["status"] == "successful") & (self._requests.index >= self._ramp_up_duration) &
                        (self._requests.index <= self._total_duration - self._ramp_down_duration)]
    if df.empty:
      return None
    df["latency_bin"] = df.apply(lambda r: int(r["latency"] // latency_bin_in_ms), axis=1)
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca(xlabel="Latency (millisec)", ylabel="Count (Requests)")
    ax.grid(alpha=0.75)
    ax.set_yscale("log")
    max_latency_in_s = (df["latency"].max() + 2 * latency_bin_in_ms) / 1000
    ax.set_xlim((0, (1000 // latency_bin_in_ms) * max_latency_in_s))
    ax.set_xticks(range(int((1000 // latency_bin_in_ms) * max_latency_in_s) + 1))
    ax.set_xticklabels(range(0, (int((1000 // latency_bin_in_ms) * max_latency_in_s) + 1) * latency_bin_in_ms, latency_bin_in_ms))
    if text_y:
      p50 = df["latency"].quantile(0.50)
      ax.axvline(x=p50 / latency_bin_in_ms, ls="dotted", lw=5, color="darkorange")
      ax.text(x=p50 / latency_bin_in_ms, y=text_y, s=" P50", fontsize=22, color="darkorange")
      p999 = df["latency"].quantile(0.999)
      ax.axvline(x=p999 / latency_bin_in_ms, ls="dotted", lw=5, color="darkorange")
      ax.text(x=p999 / latency_bin_in_ms, y=text_y, s=" P99.9", fontsize=22, color="darkorange")
    df["latency_bin"].plot(ax=ax,
                           kind="hist",
                           title="Latency Distribution of Requests",
                           bins=range(int((1000 // latency_bin_in_ms) * max_latency_in_s)),
                           grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_requests(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None, request_type=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._requests[(self._requests["status"] == "successful") & (self._requests.index >= min_time) & (self._requests.index <= max_time)]
    if request_type is not None:
      df = df[(df["type"] == request_type)]
    df = df.groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).unstack()
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, df.values.max()))
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    df.interpolate(method="linear").plot(ax=ax,
                                         kind="line",
                                         title="Instantaneous Latency of Requests" + ("" if not request_type else (" - %s" % request_type)),
                                         xlabel="Time (millisec)",
                                         ylabel="Latency (millisec)",
                                         grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_request_throughput(self, interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._requests[(self._requests.index >= min_time) & (self._requests.index <= max_time)].\
        groupby(["window_%s" % window, "status"])["window_%s" % window].count().unstack().fillna(0)
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    df.interpolate(method="linear").plot(ax=ax,
                                         kind="line",
                                         title="Request Throughput",
                                         xlabel="Time (millisec)",
                                         ylabel="Throughput (Requests/Sec)" if not interval else "Throughput (Requests/10ms)",
                                         color={
                                             "failed": "red",
                                             "successful": "blue"
                                         },
                                         legend=True,
                                         grid=True,
                                         xticks=range(int(df.index.min()),
                                                      int(df.index.max()) + 1, 60000))
    return fig

  def calculate_stats(self):
    requests = self._requests[(self._requests.index >= self._ramp_up_duration) & (self._requests.index <= self._total_duration - self._ramp_down_duration)]
    throughput = requests.groupby(["window_1000"])["window_1000"].count().\
        reindex(range(int(requests["window_1000"].min()), int(requests["window_1000"].max()) + 1, 1000), fill_value=0)
    return {
        "requests_count_total": requests.shape[0],
        "requests_count_successful": requests[requests["status"] == "successful"]["status"].count(),
        "requests_count_failed": requests[requests["status"] == "failed"]["status"].count(),
        "requests_count_read": requests[requests["rw"] == "read"]["rw"].count(),
        "requests_count_write": requests[requests["rw"] == "write"]["rw"].count(),
        "requests_ratio_successful": requests[requests["status"] == "successful"]["status"].count() / requests.shape[0],
        "requests_ratio_failed": requests[requests["status"] == "failed"]["status"].count() / requests.shape[0],
        "requests_ratio_read": requests[requests["rw"] == "read"]["rw"].count() / requests.shape[0],
        "requests_ratio_write": requests[requests["rw"] == "write"]["rw"].count() / requests.shape[0],
        "requests_latency_p50": requests[requests["status"] == "successful"]["latency"].quantile(0.50),
        "requests_latency_p95": requests[requests["status"] == "successful"]["latency"].quantile(0.95),
        "requests_latency_p99": requests[requests["status"] == "successful"]["latency"].quantile(0.99),
        "requests_latency_p999": requests[requests["status"] == "successful"]["latency"].quantile(0.999),
        "requests_latency_avg": requests[requests["status"] == "successful"]["latency"].mean(),
        "requests_latency_std": requests[requests["status"] == "successful"]["latency"].std(),
        "requests_latency_max": requests[requests["status"] == "successful"]["latency"].max(),
        "requests_throughput_p50": throughput.quantile(0.50),
        "requests_throughput_p95": throughput.quantile(0.95),
        "requests_throughput_p99": throughput.quantile(0.99),
        "requests_throughput_p999": throughput.quantile(0.999),
        "requests_throughput_avg": throughput.mean(),
        "requests_throughput_std": throughput.std(),
        "requests_throughput_max": throughput.max(),
    }


class CollectlCPULogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._cpu = build_collectl_cpu_df(experiment_dirpath)
    self._cpu_cores = {node_name: get_node_vcpus(experiment_dirpath, node_name) or range(128) for node_name in self._node_names}

  @LogAnalysis.save_fig
  def plot_cpu_metric(self, cpu_metric="total", interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = None
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or sorted(self._node_names)):
      # Data frame
      df = self._cpu[(self._cpu["node_name"] == node_name) & (self._cpu["hw_no"].isin(self._cpu_cores[node_name])) &
          (self._cpu.index >= min_time) & (self._cpu.index <= max_time)].\
          groupby(["timestamp" if not window else ("window_%s" % window), "hw_no"])[cpu_metric].mean().unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration if not window else (self._ramp_up_duration * 1000), ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) if not window else ((self._total_duration - self._ramp_down_duration) * 1000),
                 ls="--",
                 color="green")
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.set_ylim((0, 100))
      ax.grid(alpha=0.75)
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - CPU Utilization" % (node_name, self._node_labels[node_name]),
                                           xlabel=("Time (%s)" % ("sec" if not window else "millisec")) if not short else "",
                                           ylabel="%s (%%)" % cpu_metric,
                                           grid=True,
                                           legend=False,
                                           yticks=range(0, 101, 10))
    return fig

  @LogAnalysis.save_fig
  def plot_cpu_metric_comparison(self, cpu_metric="total", interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = None
      (min_time, max_time) = interval
    # Data frame
    cpu = self._cpu[(self._cpu.index >= min_time) & (self._cpu.index <= max_time)]
    cpu["node_label"] = cpu.apply(lambda r: self._node_labels[r["node_name"]], axis=1)
    df = pd.concat([cpu[(cpu["node_name"] == node_name) & (cpu["hw_no"].isin(self._cpu_cores[node_name]))] for node_name in self._node_names]).\
        groupby(["timestamp" if not window else ("window_%s" % window), "node_label"])[cpu_metric].mean().unstack()
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration if not window else (self._ramp_up_duration * 1000), ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) if not window else ((self._total_duration - self._ramp_down_duration) * 1000),
               ls="--",
               color="green")
    ax.set_xlim((df.index.min(), df.index.max()))
    ax.set_ylim((0, 100))
    ax.grid(alpha=0.75)
    df.interpolate(method="linear").plot(ax=ax,
                                         kind="line",
                                         title="CPU Utilization",
                                         xlabel="Time (%s)" % ("sec" if not window else "millisec"),
                                         ylabel="%s (%%)" % cpu_metric,
                                         grid=True,
                                         yticks=range(0, 101, 10))
    return fig


class CollectlDskLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._dsk = build_collectl_dsk_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_dsk_metric(self, dsk_metric="writes", interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = None
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or sorted(self._node_names)):
      # Data frame
      df = self._dsk[(self._dsk["node_name"] == node_name) & (self._dsk.index >= min_time) & (self._dsk.index <= max_time)].\
          groupby(["timestamp" if not window else ("window_%s" % window), "hw_no"])[dsk_metric].mean().unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration if not window else (self._ramp_up_duration * 1000), ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) if not window else ((self._total_duration - self._ramp_down_duration) * 1000),
                 ls="--",
                 color="green")
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.grid(alpha=0.75)
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - Disk I/O Utilization" % (node_name, self._node_labels[node_name]),
                                           xlabel=("Time (%s)" % ("sec" if not window else "millisec")) if not short else "",
                                           ylabel=dsk_metric,
                                           grid=True,
                                           legend=False)
    return fig

  @LogAnalysis.save_fig
  def plot_dsk_metric_comparison(self, dsk_metric="writes", interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = None
      (min_time, max_time) = interval
    # Data frame
    dsk = self._dsk[(self._dsk.index >= min_time) & (self._dsk.index <= max_time)]
    dsk["node_label"] = dsk.apply(lambda r: self._node_labels[r["node_name"]], axis=1)
    df = dsk.groupby(["timestamp" if not window else ("window_%s" % window), "node_label"])[dsk_metric].mean().unstack()
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration if not window else (self._ramp_up_duration * 1000), ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) if not window else ((self._total_duration - self._ramp_down_duration) * 1000),
               ls="--",
               color="green")
    ax.set_xlim((df.index.min(), df.index.max()))
    ax.set_ylim((0, np.nanmax(df)))
    ax.grid(alpha=0.75)
    df.interpolate(method="linear").plot(ax=ax,
                                         kind="line",
                                         title="Disk I/O Utilization",
                                         xlabel="Time (%s)" % ("sec" if not window else "millisec"),
                                         ylabel=dsk_metric,
                                         grid=True)
    return fig


class CollectlMemLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._mem = build_collectl_mem_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_mem_metric(self, mem_metric="free", interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = None
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or sorted(self._node_names)):
      # Data frame
      df = self._mem[(self._mem["node_name"] == node_name) & (self._mem.index >= min_time) & (self._mem.index <= max_time)].\
          groupby(["timestamp" if not window else ("window_%s" % window)])[mem_metric].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration if not window else (self._ramp_up_duration * 1000), ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) if not window else ((self._total_duration - self._ramp_down_duration) * 1000),
                 ls="--",
                 color="green")
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.grid(alpha=0.75)
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - Memory Utilization" % (node_name, self._node_labels[node_name]),
                                           xlabel=("Time (%s)" % ("sec" if not window else "millisec")) if not short else "",
                                           ylabel=mem_metric,
                                           grid=True,
                                           legend=False)
    return fig

  @LogAnalysis.save_fig
  def plot_mem_metric_comparison(self, mem_metric="free", interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = None
      (min_time, max_time) = interval
    # Data frame
    mem = self._mem[(self._mem.index >= min_time) & (self._mem.index <= max_time)]
    mem["node_label"] = mem.apply(lambda r: self._node_labels[r["node_name"]], axis=1)
    df = mem.groupby(["timestamp" if not window else ("window_%s" % window), "node_label"])[mem_metric].mean().unstack()
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration if not window else (self._ramp_up_duration * 1000), ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) if not window else ((self._total_duration - self._ramp_down_duration) * 1000),
               ls="--",
               color="green")
    ax.set_xlim((df.index.min(), df.index.max()))
    ax.set_ylim((0, np.nanmax(df)))
    ax.grid(alpha=0.75)
    df.interpolate(method="linear").plot(ax=ax,
                                         kind="line",
                                         title="Memory Utilization",
                                         xlabel="Time (%s)" % ("sec" if not window else "millisec"),
                                         ylabel=mem_metric,
                                         grid=True)
    return fig


class QueryLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._query = build_query_df(experiment_dirpath)
    self._dbnames = sorted(self._query["dbname"].unique())

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_queries(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None, dbnames=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(dbnames or self._dbnames) * (12 if not short else 4)))
    for (i, dbname) in enumerate(dbnames or self._dbnames):
      # Data frame
      df = self._query[(self._query["dbname"] == dbname) & (self._query.index >= min_time) & (self._query.index <= max_time)].\
          groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(dbnames or self._dbnames), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="Instantaneous Latency of Queries - %s" % dbname,
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_queries_comparison(self, latency_percentile=0.99, interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._query[(self._query.index >= min_time) & (self._query.index <= max_time)].groupby(["window_%s" % window, "dbname"])["latency"].\
        quantile(latency_percentile).unstack()
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, np.nanmax(df)))
    df.interpolate(method="linear").plot(ax=ax,
                                         kind="line",
                                         title="Instantaneous P%s Latency of Queries" % int(latency_percentile * 100),
                                         xlabel="Time (millisec)",
                                         ylabel="Latency (millisec)",
                                         grid=True)
    return fig


class QueryConnLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._query_conn = build_query_conn_df(experiment_dirpath)
    self._dbnames = sorted(self._query_conn["dbname"].unique())

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_database_connections(self,
                                                         latency_percentiles=[0.50, 0.95, 0.99, 0.999],
                                                         interval=None,
                                                         lservice=None,
                                                         dbnames=None,
                                                         short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(dbnames or self._dbnames) * (12 if not short else 4)))
    for (i, dbname) in enumerate(dbnames or self._dbnames):
      # Data frame
      df = self._query_conn[(self._query_conn.index >= min_time) & (self._query_conn.index <= max_time) & (self._query_conn["dbname"] == dbname)]
      if lservice is not None:
        df = df[df["lservice"] == lservice]
      df = df.groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(dbnames or self._dbnames), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="Instantaneous Latency of Database Connection - %s" % dbname,
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           grid=True)
    return fig


class QueryCallLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._query_call = build_query_call_df(experiment_dirpath)
    self._dbnames = sorted(self._query_call["dbname"].unique())

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_database_query_calls(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None, dbnames=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(dbnames or self._dbnames) * (12 if not short else 4)))
    for (i, dbname) in enumerate(dbnames or self._dbnames):
      # Data frame
      df = self._query_call[(self._query_call.index >= min_time) & (self._query_call.index <= max_time) & (self._query_call["dbname"] == dbname)].\
          groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(dbnames or self._dbnames), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="Instantaneous Latency of Database Query Calls - %s" % dbname,
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_database_query_calls_comparison(self, latency_percentile=0.99, interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._query_call[(self._query_call.index >= min_time) & (self._query_call.index <= max_time)].\
        groupby(["window_%s" % window, "dbname"])["latency"].quantile(latency_percentile).unstack()
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, np.nanmax(df)))
    df.interpolate(method="linear").plot(ax=ax,
                                         kind="line",
                                         title="Instantaneous P%s Latency of Database Query Calls" % int(latency_percentile * 100),
                                         xlabel="Time (millisec)",
                                         ylabel="Latency (millisec)",
                                         grid=True)
    return fig


class RedisLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._redis = build_redis_df(experiment_dirpath)
    self._keys = sorted(self._redis["key"].unique())

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_commands(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(self._keys) * 12))
    for (i, key) in enumerate(self._keys):
      # Data frame
      df = self._redis[(self._redis["key"] == key) & (self._redis.index >= min_time) & (self._redis.index <= max_time)].\
          groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(self._keys), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="Instantaneous Latency of Commands - %s" % key,
                                           xlabel="Time (millisec)",
                                           ylabel="Latency (millisec)",
                                           grid=True)
    return fig


class RPCLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._rpc = build_rpc_df(experiment_dirpath)
    self._function_names = self._rpc["rfunction"].unique()

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_rpcs(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None, lservice=None, rfunctions=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(rfunctions or self._function_names) * (12 if not short else 4)))
    for (i, function) in enumerate(rfunctions or self._function_names):
      # Data frame
      df = self._rpc[(self._rpc["rfunction"] == function) & (self._rpc.index >= min_time) & (self._rpc.index <= max_time)].\
          groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(rfunctions or self._function_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="Instantaneous Latency of RPC - %s" % function,
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_rpcs_comparison(self, latency_percentile=0.99, interval=None, rfunctions=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._rpc[(self._rpc.index >= min_time) & (self._rpc.index <= max_time)]
    if rfunctions:
      df = df[df["rfunction"].isin(rfunctions)]
    df = df.groupby(["window_%s" % window, "rfunction"])["latency"].quantile(latency_percentile).unstack()
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, np.nanmax(df)))
    df.interpolate(method="linear").plot(ax=ax,
                                         kind="line",
                                         title="Instantaneous P%s Latency of RPC" % int(latency_percentile * 100),
                                         xlabel="Time (millisec)",
                                         ylabel="Latency (millisec)",
                                         grid=True)
    return fig


class RPCConnLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._rpc_conn = build_rpc_conn_df(experiment_dirpath)
    self._service_names = self._rpc_conn["rservice"].unique()

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_service_connections(self,
                                                        latency_percentiles=[0.50, 0.95, 0.99, 0.999],
                                                        interval=None,
                                                        lservice=None,
                                                        rservices=None,
                                                        short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(rservices or self._service_names) * (12 if not short else 4)))
    for (i, service) in enumerate(rservices or self._service_names):
      # Data frame
      df = self._rpc_conn[(self._rpc_conn.index >= min_time) & (self._rpc_conn.index <= max_time) & (self._rpc_conn["rservice"] == service)]
      if lservice is not None:
        df = df[df["lservice"] == lservice]
      df = df.groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(rservices or self._service_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="Instantaneous Latency of Service Connection - %s" % service,
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           grid=True)
    return fig


class RPCCallLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._rpc_call = build_rpc_call_df(experiment_dirpath)
    self._function_names = self._rpc_call["rfunction"].unique()

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_rpc_calls(self, latency_percentiles=[0.50, 0.95, 0.99, 0.999], interval=None, lservice=None, rfunctions=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(rfunctions or self._function_names) * (12 if not short else 4)))
    for (i, function) in enumerate(rfunctions or self._function_names):
      # Data frame
      df = self._rpc_call[(self._rpc_call.index >= min_time) & (self._rpc_call.index <= max_time) & (self._rpc_call["rfunction"] == function)]
      if lservice is not None:
        df = df[df["lservice"] == lservice]
      df = df.groupby(["window_%s" % window])["latency"].quantile(latency_percentiles).unstack()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(rfunctions or self._function_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="Instantaneous Latency of RPC Calls - %s" % function,
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_instantaneous_latency_of_rpc_calls_comparison(self, latency_percentile=0.99, interval=None, rfunctions=None):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    # Data frame
    df = self._rpc_call[(self._rpc_call.index >= min_time) & (self._rpc_call.index <= max_time)]
    if rfunctions:
      df = df[df["rfunction"].isin(rfunctions)]
    df = df.groupby(["window_%s" % window, "rfunction"])["latency"].quantile(latency_percentile).unstack()
    if df.empty:
      return None
    # Plot
    fig = plt.figure(figsize=(24, 12))
    ax = fig.gca()
    ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
    ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
    ax.grid(alpha=0.75)
    ax.set_xlim((min_time * 1000, max_time * 1000))
    ax.set_ylim((0, np.nanmax(df)))
    df.interpolate(method="linear").plot(ax=ax,
                                         kind="line",
                                         title="Instantaneous P%s Latency of RPC Calls" % int(latency_percentile * 100),
                                         xlabel="Time (millisec)",
                                         ylabel="Latency (millisec)",
                                         grid=True)
    return fig


class CPURunQLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._cpurunq = build_cpurunq_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_cpu_run_queue_length(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._cpurunq[(self._cpurunq["node_name"] == node_name) & (self._cpurunq.index >= min_time) & (self._cpurunq.index <= max_time)].\
          groupby(["window_%s" % window])["qlen_max"].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - CPU Run Queue Length" % (node_name, self._node_labels[node_name]),
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Count (Tasks)",
                                           color="black",
                                           grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_cpu_run_queue_latency(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._cpurunq[(self._cpurunq["node_name"] == node_name) & (self._cpurunq.index >= min_time) & (self._cpurunq.index <= max_time)].\
          groupby(["window_%s" % window])["qlat_max"].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - CPU Run Queue Latency" % (node_name, self._node_labels[node_name]),
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           color="black",
                                           grid=True)
    return fig


class CPURunQFuncLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._cpurunqfunc = build_cpurunqfunc_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_cpu_run_queue_length(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._cpurunqfunc[(self._cpurunqfunc["node_name"] == node_name) & (self._cpurunqfunc.index >= min_time) &
                             (self._cpurunqfunc.index <= max_time)].groupby(["window_%s" % window])["qlen_max"].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - CPU Run Queue Length" % (node_name, self._node_labels[node_name]),
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Count (Tasks)",
                                           color="black",
                                           grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_cpu_run_queue_latency(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._cpurunqfunc[(self._cpurunqfunc["node_name"] == node_name) & (self._cpurunqfunc.index >= min_time) &
                             (self._cpurunqfunc.index <= max_time)].groupby(["window_%s" % window])["qlat_max"].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - CPU Run Queue Latency" % (node_name, self._node_labels[node_name]),
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           color="black",
                                           grid=True)
    return fig


class TCPSynblLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._synbl = build_tcpsynbl_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_syn_backlog_length(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._synbl[(self._synbl["node_name"] == node_name) & (self._synbl.index >= min_time) & (self._synbl.index <= max_time)].groupby(
          ["window_%s" % window])["synbl_len_max"].max()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - TCP SYN Backlog Length" % (node_name, self._node_labels[node_name]),
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Count (Requests)",
                                           color="black",
                                           grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_syn_backlog_latency(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._synbl[(self._synbl["node_name"] == node_name) & (self._synbl.index >= min_time) & (self._synbl.index <= max_time)].\
          groupby(["window_%s" % window])["synbl_lat_max"].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - TCP SYN Backlog Latency" % (node_name, self._node_labels[node_name]),
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           color="black",
                                           grid=True)
    return fig


class TCPAcceptqLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._acceptq = build_tcpacceptq_df(experiment_dirpath)

  @LogAnalysis.save_fig
  def plot_accept_queue_length(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._acceptq[(self._acceptq["node_name"] == node_name) & (self._acceptq.index >= min_time) & (self._acceptq.index <= max_time)].\
          groupby(["window_%s" % window])["acceptq_len_max"].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - TCP Accept Queue Length" % (node_name, self._node_labels[node_name]),
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Count (Requests)",
                                           color="black",
                                           grid=True)
    return fig

  @LogAnalysis.save_fig
  def plot_accept_queue_latency(self, interval=None, node_names=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(node_names or self._node_names) * (12 if not short else 4)))
    for (i, node_name) in enumerate(node_names or self._node_names):
      # Data frame
      df = self._acceptq[(self._acceptq["node_name"] == node_name) & (self._acceptq.index >= min_time) & (self._acceptq.index <= max_time)].\
          groupby(["window_%s" % window])["acceptq_lat_max"].mean()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(node_names or self._node_names), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((min_time * 1000, max_time * 1000))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s: %s - TCP Accept Queue Latency" % (node_name, self._node_labels[node_name]),
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Latency (millisec)",
                                           color="black",
                                           grid=True)
    return fig


class TCPRetransLogAnalysis(LogAnalysis):

  def __init__(self, experiment_dirpath, output_dirpath=None):
    super().__init__(experiment_dirpath, output_dirpath)
    self._retrans = build_tcpretrans_df(experiment_dirpath)
    self._addr_port = [(addr, port) for addr, port in set(zip(self._retrans["raddr"], self._retrans["rport"])) if port > 1080 and port < 1100]

  def plot_number_of_tcp_packet_retransmissions(self, interval=None, addr_port=None, short=False):
    if not interval:
      window = 1000
      min_time = 0
      max_time = self._total_duration
    else:
      window = 10
      (min_time, max_time) = interval
    fig = plt.figure(figsize=(24, len(addr_port or self._addr_port) * (12 if not short else 4)))
    for (i, ap) in enumerate(addr_port or self._addr_port):
      # Data frame
      df = self._retrans[(self._retrans["raddr"] == ap[0]) & (self._retrans["rport"] == ap[1]) & (self._retrans.index >= min_time) &
                         (self._retrans.index <= max_time)].groupby(["window_%s" % window])["window_%s" % window].count()
      if df.empty:
        continue
      # Plot
      ax = fig.add_subplot(len(addr_port or self._addr_port), 1, i + 1)
      ax.axvline(x=self._ramp_up_duration * 1000, ls="--", color="green")
      ax.axvline(x=(self._total_duration - self._ramp_down_duration) * 1000, ls="--", color="green")
      ax.grid(alpha=0.75)
      ax.set_xlim((df.index.min(), df.index.max()))
      ax.set_ylim((0, df.values.max()))
      df.interpolate(method="linear").plot(ax=ax,
                                           kind="line",
                                           title="%s:%s - TCP Packet Retransmissions" % (ap[0], ap[1]),
                                           xlabel="Time (millisec)" if not short else "",
                                           ylabel="Count (Retransmissions)",
                                           color="black",
                                           grid=True)
    return fig


def main():
  # Parse command-line arguments.
  parser = argparse.ArgumentParser(description="Plot experiment graphs")
  parser.add_argument("--experiment_dirname",
                      required=False,
                      action="store",
                      type=str,
                      help="Name of directory containing experiment data in `../data`",
                      default="")
  parser.add_argument("--distribution", action="store_true", default=False, help="Add distribution graphs")
  args = parser.parse_args()
  # List experiment(s) directory.
  experiment_dirpaths = [
      os.path.join(os.path.abspath(""), "..", "data", dirname)
      for dirname in ([args.experiment_dirname] if args.experiment_dirname else os.listdir(os.path.join(os.path.abspath(""), "..", "data")))
      if re.findall("BuzzBlogBenchmark_", dirname) and not re.findall(".tar.gz", dirname)
  ]
  # Retrieve list of experiments whose graphs have already been plotted.
  plotted_dirnames = []
  try:
    os.mkdir(os.path.join(os.path.abspath(""), "..", "graphs"))
  except FileExistsError:
    plotted_dirnames = os.listdir(os.path.join(os.path.abspath(""), "..", "graphs"))
  # Plot experiment graphs.
  for experiment_dirpath in experiment_dirpaths:
    if os.path.basename(experiment_dirpath) in plotted_dirnames:
      continue
    print("Processing %s:" % experiment_dirpath)
    output_dirpath = os.path.join(os.path.abspath(""), "..", "graphs", os.path.basename(experiment_dirpath))
    os.mkdir(output_dirpath)
    for notebook_cls in [
        RequestLogAnalysis, CollectlCPULogAnalysis, CollectlDskLogAnalysis, CollectlMemLogAnalysis, QueryLogAnalysis, QueryConnLogAnalysis,
        QueryCallLogAnalysis, RPCLogAnalysis, RPCConnLogAnalysis, RPCCallLogAnalysis, RedisLogAnalysis, CPURunQLogAnalysis, CPURunQFuncLogAnalysis,
        TCPSynblLogAnalysis, TCPAcceptqLogAnalysis, TCPRetransLogAnalysis
    ]:
      try:
        notebook = notebook_cls(experiment_dirpath, output_dirpath)
        notebook.plot(distribution=args.distribution)
      except Exception as e:
        print("\tFailed: %s" % str(e))
        continue


if __name__ == "__main__":
  main()

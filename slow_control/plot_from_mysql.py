from __future__ import absolute_import


import sys, os, time, calendar, signal
import MySQLdb
from array import array
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.dates import DateFormatter
import matplotlib.dates as dates
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime
import pytz
from scipy.signal import medfilt

# Ugly workaround - we want to be able to run this directly
# or import from it...
if __name__=='__main__' and __package__ is None:
    from par_config import par_map
    import credentials
else:
    from .par_config import par_map
    from . import credentials

import argparse


def datenum_to_epoch(datenum, dst=True):
    # assumes datenum is yymmddhhmmss as a string
    # TODO: fails if you do first hour of first day of month
    if '_' in datenum:
        datenum = datenum.replace('_', '')
    offset = 0
    if dst:
        offset = 1
    day = int(datenum[4:6])
    hour = (int(datenum[6:8]) - offset)
    if hour < 0:
        day -= 1
    nowtime = datetime(
        int('20'+datenum[0:2]),
        int(datenum[2:4]),
        day,
        hour % 24,
        int(datenum[8:10]),
        int(datenum[10:12]),
        tzinfo=pytz.timezone('US/Eastern')
        )
    thentime = datetime(1970, 1, 1, tzinfo=pytz.utc)
    return (nowtime - thentime).total_seconds()



def get_data_from_mysql(table, column_name, t0=1410802453, t1=int(time.time())):
    database_user = credentials.database_user
    database_pass = credentials.database_pass

    pid = os.spawnlp(os.P_NOWAIT, 'ssh', 'ssh', '-L', '3307:localhost:3306', '-N', 'xedaq@xeclipse.astro.columbia.edu', '-p', '7920')
    time.sleep(1)

    # open the connection to the database
    try:
        connection = MySQLdb.connect('127.0.0.1', database_user, database_pass, 'smac', port=3307)
        cursor = connection.cursor()
    except MySQLdb.Error, e:
        print 'problem connection to run database, error %d: %s' % (e.args[0], e.args[1])
        sys.exit(1)

    column_list = table + '_id' + ',' + column_name
    query = 'select %s from %s where (%s_id > %i) && (%s_id < %i);' % (column_list, table, table, t0, table, t1)
    cursor.execute(query)
    data = cursor.fetchall()
    connection.close()

    os.kill(pid, signal.SIGTERM)

    return data

def write_to_file(data, outfile):
    with open(outfile, 'w') as f:
        for row in data:
            if None in row:
                continue
            f.write('%i\t%.2e\n' % (row[0], row[1]))
    return

def read_from_file(infile):
    data = []
    with open(outfile, 'r') as f:
        for row in f.readlines():
            data.append(np.asarray(row.split(), dtype=float).tolist())
    return data

def plot_line(axis, datenum, ylim=[0,1800], label=None):
    datetimes = dates.date2num([datetime.fromtimestamp(int(datenum_to_epoch(datenum)))]*2)
    axis.plot_date(datetimes, ylim, 'r--', label=label)
    return

def extra_plotting(axis):
    #plot_line(axis, '180507142400')
    #plt.text(0.5, 0.75, 'Turn on\nGXe Circulation', transform=axis.transAxes,
    #    fontsize=18, color='r')
    return


def plot_directly(data, par_name, save_name=False, yscale='linear', newfig=True, show=True, ax=None, twin=False, color='k', medfilter=False, ylim=False, extra=True):
    if newfig:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
    times = [row[0] for row in data]
    values = [row[1] for row in data]
    datetimes = dates.date2num([datetime.fromtimestamp(time) for time in times])
    if medfilter:
        values = medfilt(values, kernel_size=31)
    if twin:
        ax2 = ax.twinx()
        ax2.plot_date(datetimes, values, '%s.' % color)
        if ylim:
            ax2.set_ylim(ylim)
        plt.yscale('linear')
        plt.tick_params(axis='y', which='minor')
        ax2.tick_params('y', colors=color)
        plt.ylabel(par_name + ' [' + par_map[par_name]['unit'] +']', color=color)
    else:
        ax.plot_date(datetimes, values, '%s.' % color)
        date_format = dates.DateFormatter('%Y/%m/%d\n%H:%M')
        ax.xaxis.set_major_formatter(date_format)
        if newfig:
            fig.autofmt_xdate()
        plt.yscale('linear')
        plt.tick_params(axis='y', which='minor')
        ax.tick_params('y', colors=color)
        if ylim:
            ax.set_ylim(ylim)
        plt.ylabel(par_name + ' [' + par_map[par_name]['unit'] +']', color=color)
        plt.xlabel('Time')
        plt.grid()
    if extra:
        extra_plotting(ax)
    if save_name:
        plt.savefig(save_name)
    if show:
        plt.show()
    return ax


def parse_sys_args(argv):
    ###### Load setup arguments #####
    parser = argparse.ArgumentParser(description="Reduction of XENON1T data")

    parser.add_argument('--par_name', dest='par_name', required=True,
                        action='store', type=str,
                        help='Parameter to grab from MySQL')

    parser.add_argument('--start_time', dest='start_timestr', default='20180121_000000',
                        action='store', type=str,
                        help='Start time (format YYMMDD_hhmmss)')

    parser.add_argument('--stop_time', dest='stop_timestr', default='now',
                        action='store', type=str,
                        help='Stop time (format YYMMDD_hhmmss or \"now\")')

    parser.add_argument('--second_par_name', dest='par_name_2', default=None,
                        action='store', type=str,
                        help='Parameter to grab from MySQL')

    parser.add_argument('--ymin', dest='ymin', default=None,
                        action='store', type=float,
                        help='ymin for par plot')

    parser.add_argument('--ymax', dest='ymax', default=None,
                        action='store', type=float,
                        help='ymin for par plot')

    parser.add_argument('--second_ymin', dest='second_ymin', default=None,
                        action='store', type=float,
                        help='ymin for par plot')

    parser.add_argument('--no_plot', dest='no_plot',
                        action='store_true',
                        help='Suppress all plotting')

    parser.add_argument('--data_filename', dest='data_filename', default=None,
                        action='store', type=str,
                        help='Name to store data as a two column .dat')

    parser.add_argument('--from_file', dest='infilename', default=None,
                        action='store', type=str,
                        help='File to take data from if you don\'t want to use MySQL')

    parser.add_argument('--plot_filename', dest='plot_filename', default=None,
                        action='store', type=str,
                        help='Plot filename')

    parser.add_argument('--filter_kernel', dest='filter_kernel', default=None,
                        action='store', type=int,
                        help='Kernel for median filter - must be odd int (no filter if not specified)')

    return parser.parse_args(argv)



if __name__ == '__main__':
    args = parse_sys_args(sys.argv[1:])
    ylim = None
    if (args.ymin is not None) and (args.ymax is not None):
        ylim=[args.ymin, args.ymax]
    t0 = datenum_to_epoch(args.start_timestr.replace("_", ""))
    if args.stop_timestr=='now':
        t1 = int(time.time())
    else:
        t1 = datenum_to_epoch(args.stop_timestr.replace("_", ""))

    if args.infilename:
        data = read_from_file(args.infilename)
    else:
        data = get_data_from_mysql(par_map[args.par_name]['table'], args.par_name, t0, t1)
    if args.data_filename:
        write_to_file(data, args.data_filename)
    if not args.no_plot:
        if args.par_name_2:
            ax = plot_directly(data, args.par_name, save_name=False, show=False, color='r', medfilter=args.filter_kernel, ylim=ylim)
            data_2 = get_data_from_mysql(par_map[args.par_name_2]['table'], args.par_name_2, t0, t1)
            plot_directly(data_2, args.par_name_2, newfig=False, twin=True, ax=ax, color='b', save_name=args.plot_filename, medfilter=args.filter_kernel)
        else:
            ax = plot_directly(data, args.par_name, save_name=args.plot_filename, show=True, medfilter=args.filter_kernel, ylim=ylim)

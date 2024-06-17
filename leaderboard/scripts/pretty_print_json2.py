# !/usr/bin/env python
# Copyright (c) 2020 Intel Corporation.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Create a human readable version of the scores provided by the leaderboard.
"""

from __future__ import print_function

import argparse
from argparse import RawTextHelpFormatter
from dictor import dictor
import json
from tabulate import tabulate
import pandas as pd


def prettify_json(args):
    
    json_path = f"{args.experiment_name}/evaluation/results.json"
    csv_path = f"{args.experiment_name}/evaluation/results.csv"
    
    with open(json_path) as fd:
        json_dict = json.load(fd)

    if not json_dict:
        print('[Error] The file [{}] could not be parsed.'.format(args.file))
        return -1

    # progress = dictor(json_dict, '_checkpoint.progress')
    records_table = dictor(json_dict, '_checkpoint.records')
    # sensors = dictor(json_dict, 'sensors')
    # labels_scores = dictor(json_dict, 'labels')
    # scores = dictor(json_dict, 'values')

    # compose output
    output = ""

    # global_data is the last line of the table.
    # if scores and labels_scores:
    #     global_data = list(zip(*[labels_scores, scores]))
    #     print(global_data)
        
    if records_table:
        header = ['Metric', 'Value', 'Additional information']
        list_statistics = [header]
        total_duration_game = 0
        total_duration_system = 0
        total_route_length = 0
        for route in records_table:
            
            if route['status'] == 'Failed - Simulation crashed':
                continue
            
            
            route_completed_kms = 0.01 * route['scores']['score_route'] * route['meta']['route_length'] / 1000.0
            metrics_route = [[key, '{:.4f}'.format(values), ''] for key, values in route['scores'].items()]
            infractions_route = [[key, '{:.4f}'.format(len(values)/route_completed_kms),
                                 '\n'.join(values)] for key, values in route['infractions'].items()]

            times = [['Game duration', '{:.4f}'.format(route['meta']['duration_game']), 'seconds'],
                     ['System duration', '{:.4f}'.format(route['meta']['duration_system']), 'seconds']]

            route_completed_length = [ ['distance driven', '{:.4f}'.format(route_completed_kms), 'Km']]

            total_duration_game += route['meta']['duration_game']
            total_duration_system += route['meta']['duration_system']
            total_route_length += route_completed_kms

            list_statistics.extend([['{}'.format(route['route_id']), '', '']])
            list_statistics.extend([*metrics_route, *infractions_route, *times, *route_completed_length])
            # list_statistics.extend([['', '', '']])

        # list_statistics.extend([['Total game duration', '{:.4f}'.format(total_duration_game), 'seconds']])
        # list_statistics.extend([['Total system duration', '{:.4f}'.format(total_duration_system), 'seconds']])
        # list_statistics.extend([['Total distance driven', '{:.4f}'.format(total_route_length), 'Km']])

        output += '==== Per-route analysis: ===\n'.format()
        

        # print(list_statistics)
        scenario_data = {}
        current_scenario = None
        # exit(0)

        for item in list_statistics:
            if item[0].startswith('RouteScenario'):
                # New scenario encountered
                current_scenario = item[0]
                scenario_data[current_scenario] = {}
            elif current_scenario:
                # Add metric to the current scenario
                scenario_data[current_scenario][item[0]] = item[1]

        # Convert the populated dictionary to a pandas DataFrame
        df = pd.DataFrame.from_dict(scenario_data, orient='index')

        df.to_csv(csv_path)



    # if args.output:
    #     with open(args.output, 'w') as fd:
    #         fd.write(output)
    # else:
    #     print(output)

    return 0


def main():
    description = 'Create a human readable version of the scores provided by the leaderboard.\n'
    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-en', '--experiment_name', help='JSON file containing the results of the leaderboard', required=True)
    parser.add_argument('--format', default='fancy_grid',
                        help='Format in which the table will be printed, e.g.: fancy_grid, latex, github, html, jira')
    arguments = parser.parse_args()

    return prettify_json(arguments)


if __name__ == '__main__':
    main()

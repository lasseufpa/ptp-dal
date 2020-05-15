import unittest
import copy
from datetime import datetime
from collections import OrderedDict
from manager import Formatter

metadata_v1 = {
    'oscillator': 'xo',
    'sync_period': 0.0625,
    'fh_traffic': {
        'type': 'inline',
        'fs': 76,
        'bitrate': {
            'dl': 83,
            'ul': 84
        },
        'iq_size': 24,
        'n_spf': 64,
        'n_rru_dl': 2,
        'n_rru_ul': 2,
        'vlan_pcp': 1
    },
    'delay_cal': False,
    'hops': 4,
    'n_rru_ptp': 2,
    'pipelines': {
        "bbu" : 239,
        "rru" : 239
    },
    'start_time': '2020-04-25 09:47:07'
}

metadata_v2 = {
    'oscillator': 'xo',
    'sync_period': 0.0625,
    'fh_traffic': {
        'type': 'inline',
        'fs': 76,
        'bitrate': {
            'dl': 83,
            'ul': 84
        },
        'iq_size': {
            'dl': 24,
            'ul': 24
        },
        'n_spf': {
            'dl': 64,
            'ul': 64,
        },
        'n_rru_dl': 2,
        'n_rru_ul': 2,
        'vlan_pcp': 1
    },
    'delay_cal': False,
    'hops': 4,
    'n_rru_ptp': 2,
    'pipelines': {
        "bbu" : 239,
        "rru" : 239
    },
    'start_time': '2020-04-25 09:47:07'
}

expected_md_structure = {
    'name' : 'test-name',
    'oscillator': 'xo',
    'sync_period': 0.0625,
    'fh_type': 'inline',
    'fh_fs': 76,
    'fh_bitrate_dl': 83,
    'fh_bitrate_ul': 84,
    'fh_iq_size_dl': 24,
    'fh_iq_size_ul': 24,
    'fh_n_spf_dl': 64,
    'fh_n_spf_ul': 64,
    'fh_n_rru_dl': 2,
    'fh_n_rru_ul': 2,
    'fh_vlan_pcp': 1,
    'delay_cal': False,
    'delay_cal_duration': None,
    'hops_rru1': 4,
    'hops_rru2': 4,
    'n_rru_ptp': 2,
    'pipeline_bbu': 239,
    'pipeline_rru': 239,
    'start_time': datetime.strptime('2020-04-25 09:47:07', '%Y-%m-%d %H:%M:%S'),
    'fh_traffic': True
}


class TestFormatter(unittest.TestCase):
    def test_formatter_v1(self):
        formatter   = Formatter('test-name', metadata_v1)
        formated_md = formatter.format()
        self.assertEqual(expected_md_structure, formated_md)

    def test_formatter_v2(self):
        formatter   = Formatter('test-name', metadata_v2)
        formated_md = formatter.format()
        self.assertEqual(expected_md_structure, formated_md)


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
    'delay_cal': True,
    'delay_cal_duration': 300,
    'hops': 4,
    'n_rru_ptp': 2,
    'pipelines': {
        "bbu" : 239,
        "rru" : 239
    },
    'ptp_unicast': False,
    'departure_ctrl': False,
    'departure_gap': None,
    'tstamp_latency_corr': {
        'bbu': {
            'tx': 80,
            'rx': 80,
        },
        'rru1': {
            'tx': 80,
            'rx': 80,
        },
        'rru2': {
            'tx': 80,
            'rx': 80
        }
    },
    'n_exchanges': 1000,
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
    'calibration': True,
    'calibration_duration': 300,
    'hops': 4,
    'n_rru_ptp': 2,
    'pipelines': {
        "bbu" : 239,
        "rru" : 239
    },
    'ptp_unicast': False,
    'departure_ctrl': False,
    'departure_gap': None,
    'tstamp_latency_corr': {
        'bbu': {
            'tx': 80,
            'rx': 80,
        },
        'rru1': {
            'tx': 80,
            'rx': 80,
        },
        'rru2': {
            'tx': 80,
            'rx': 80
        }
    },
    'n_exchanges': 1000,
    'start_time': '2020-04-25 09:47:07'
}

metadata_v3 = {
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
    'bg_traffic': {
        'type': 'cross',
        'cross_info': {
            'flow': 'bi',
            'model': 2,
            'bw': 75
        }
    },
    'delay_cal': True,
    'delay_cal_duration': 300,
    'hops': 4,
    'n_rru_ptp': 2,
    'pipelines': {
        "bbu" : 239,
        "rru" : 239
    },
    'ptp_unicast': False,
    'departure_ctrl': False,
    'departure_gap': {},
    'tstamp_latency_corr': {
        'bbu': {
            'tx': 80,
            'rx': 80,
        },
        'rru1': {
            'tx': 80,
            'rx': 80,
        },
        'rru2': {
            'tx': 80,
            'rx': 80
        }
    },
    'n_exchanges': 1000,
    'start_time': '2020-04-25 09:47:07'
}

class TestFormatter(unittest.TestCase):
    def setUp(self):
        self.expected_md_structure = {
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
            'bg_traffic_type': None,
            'bg_traffic_cross_info_flow': None,
            'bg_traffic_cross_info_model': None,
            'bg_traffic_cross_info_bw': None,
            'calibration': True,
            'calibration_duration': 300,
            'hops_rru1': 4,
            'hops_rru2': 4,
            'n_rru_ptp': 2,
            'pipeline_bbu': 239,
            'pipeline_rru': 239,
            'start_time': datetime.strptime('2020-04-25 09:47:07', '%Y-%m-%d %H:%M:%S'),
            'departure_ctrl': False,
            'departure_gap_dl': None,
            'departure_gap_ul': None,
            'ptp_unicast': False,
            'tstamp_latency_corr_bbu_tx': 80,
            'tstamp_latency_corr_bbu_rx': 80,
            'tstamp_latency_corr_rru1_tx': 80,
            'tstamp_latency_corr_rru1_rx': 80,
            'tstamp_latency_corr_rru2_tx': 80,
            'tstamp_latency_corr_rru2_rx': 80,
            'fh_traffic': True,
            'bg_traffic': False,
            'n_exchanges': 1000
        }

    def test_formatter_v1(self):
        formatter   = Formatter('test-name', metadata_v1)
        formated_md = formatter.format()
        self.assertEqual(self.expected_md_structure, formated_md)

    def test_formatter_v2(self):
        formatter   = Formatter('test-name', metadata_v2)
        formated_md = formatter.format()
        self.assertEqual(self.expected_md_structure, formated_md)

    def test_formatter_v3(self):
        formatter   = Formatter('test-name', metadata_v3)
        formated_md = formatter.format()

        # Add the bg-traffic fields to the expected structure
        self.expected_md_structure['bg_traffic']                  = True
        self.expected_md_structure['bg_traffic_type']             = 'cross'
        self.expected_md_structure['bg_traffic_cross_info_flow']  = 'bi'
        self.expected_md_structure['bg_traffic_cross_info_model'] = 2
        self.expected_md_structure['bg_traffic_cross_info_bw']    = 75

        self.assertEqual(self.expected_md_structure, formated_md)



import unittest
import copy
from datetime import datetime
from collections import OrderedDict
from manager import Formatter


class TestFormatter(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            'oscillator': 'xo',
            'sync_period': 0.0625,
            'fh_traffic': {
                'type': 'inline',
                'fs': 76,
                'bitrate': 83,
                'iq_size': 24,
                'n_spf': 64,
                'n_rru_dl': 2,
                'n_rru_ul': 2,
                'vlan_pcp': 1
            },
            'bg_traffic': None,
            'delay_cal': False,
            'delay_cal_duration': 0,
            'hops': 4,
            'n_rru_ptp': 2,
            'pipelines': {
                'bbu': 239,
                'rru': 239
            },
            'ptp_unicast': False,
            'departure_ctrl': False,
            'departure_gap': {},
            'tstamp_latency_corr': {
                'bbu' : 80,
                'rru1': 80,
                'rru2': 80
            },
            'n_exchanges': 1000,
            'start_time': '2020-04-25 09:47:07'
        }

        # Expected metadata structure that should be returned from the
        # formatter class when passing the metadata above.
        self.expected_md_structure = {
            'name' : 'test-name',
            'oscillator': 'xo',
            'sync_period': 0.0625,
            'fh_traffic': True,
            'fh_type': 'inline',
            'fh_fs': 76,
            'fh_bitrate_dl': 83,
            'fh_bitrate_ul': 83,
            'fh_iq_size_dl': 24,
            'fh_iq_size_ul': 24,
            'fh_n_spf_dl': 64,
            'fh_n_spf_ul': 64,
            'fh_n_rru_dl': 2,
            'fh_n_rru_ul': 2,
            'fh_vlan_pcp': 1,
            'bg_traffic': False,
            'bg_traffic_type': None,
            'bg_traffic_cross_info_flow': None,
            'bg_traffic_cross_info_model': None,
            'bg_traffic_cross_info_bw': None,
            'calibration': False,
            'calibration_duration': 0,
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
            'n_exchanges': 1000
        }

    def test_formatter(self):
        """Test the general formatter class
        """
        formatter    = Formatter('test-name', self.metadata)
        formatter_md = formatter.format()
        self.assertEqual(self.expected_md_structure, formatter_md)

    def test_distinct_values(self):
        """Test fields that support distinct values for dl and ul
        """
        self.metadata['fh_traffic']['iq_size'] = {
            'dl': 24,
            'ul': 23
        }
        self.metadata['fh_traffic']['n_spf'] = {
            'dl': 64,
            'ul': 63
        }

        formatter   = Formatter('test-name', self.metadata)
        formated_md = formatter.format()

        # Update the values on the expected metadata
        self.expected_md_structure['fh_iq_size_dl'] = 24
        self.expected_md_structure['fh_iq_size_ul'] = 23
        self.expected_md_structure['fh_n_spf_dl']   = 64
        self.expected_md_structure['fh_n_spf_ul']   = 63
        self.assertEqual(self.expected_md_structure, formated_md)

    def test_md_without_fh_traffic(self):
        """Test metadata without FH traffic
        """
        # Remove fh_traffic values from the self.metadata
        self.metadata['fh_traffic'] = None

        formatter   = Formatter('test-name', self.metadata)
        formated_md = formatter.format()

        # Update the values on the expected metadata
        for k in self.expected_md_structure.keys():
            if 'fh_' in k:
                self.expected_md_structure[k] = None
        self.expected_md_structure['fh_traffic'] = False

        self.assertEqual(self.expected_md_structure, formated_md)

    def test_md_with_bg_traffic(self):
        """Test metadata with background traffic
        """
        # Update the metadata to include BG traffic information
        self.metadata['bg_traffic'] = {
            'type': 'cross',
            'cross_info': {
                'flow': 'bi',
                'model': 2,
                'bw': 75
            }
        }

        formatter   = Formatter('test-name', self.metadata)
        formated_md = formatter.format()

        # Add the bg-traffic fields to the expected structure
        self.expected_md_structure['bg_traffic']                  = True
        self.expected_md_structure['bg_traffic_type']             = 'cross'
        self.expected_md_structure['bg_traffic_cross_info_flow']  = 'bi'
        self.expected_md_structure['bg_traffic_cross_info_model'] = 2
        self.expected_md_structure['bg_traffic_cross_info_bw']    = 75

        self.assertEqual(self.expected_md_structure, formated_md)



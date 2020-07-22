import os, json, logging
from flask import send_file, request
from flask_restful import Resource, reqparse
from manager import Manager


class DatasetSearch(Resource):
    def __init__(self, app):
        self.app = app

    def post(self):
        manager = Manager(self.app)
        query   = manager.search(request.json)
        results = list()

        if (len(query) > 0):
            for ds in query:
                results.append({
                    'name'               : ds.name,
                    'oscillator'         : ds.oscillator,
                    'sync-period'        : ds.sync_period,
                    'fh-traffic'         : ds.fh_traffic,
                    'calibration'        : ds.delay_cal,
                    'departure-ctrl'     : ds.departure_ctrl,
                    'n-rrus (dl/ul)'     : f'{ds.fh_n_rru_dl}/{ds.fh_n_rru_ul}',
                    'n-rru-ptp'          : ds.n_rru_ptp,
                    'n-hops (rru1/rru2)' : f'{ds.hops_rru1}/{ds.hops_rru2}',
                    'pipeline (bbu/rru)' : f'{ds.pipeline_bbu}/{ds.pipeline_rru}',
                    'start-time'         : ds.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'n-exchanges'        : ds.n_exchanges
                })
            return {'found': results}
        else:
            return {'found': 'No dataset found!'}, 404


class DatasetDownload(Resource):
    def get(self, dataset):
        ds_folder   = "/app/api/datasets/"
        ds_filename = os.path.join(ds_folder, dataset)

        if (os.path.isfile(ds_filename)):
            return send_file(ds_filename, as_attachment=True)
        else:
            return {'message': 'Dataset not found!'}, 404


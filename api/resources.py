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
                    'name'         : ds.name,
                    'oscillator'   : ds.oscillator,
                    'sync-period'  : ds.sync_period,
                    'calibration'  : ds.delay_cal,
                    'cal-duration' : ds.delay_cal_duration,
                    'fh-traffic'   : ds.fh_traffic,
                    'n-rrus-dl'    : ds.fh_n_rru_dl,
                    'n-rrus-ul'    : ds.fh_n_rru_ul,
                    'n-rru-ptp'    : ds.n_rru_ptp,
                    'hops-rru1'    : ds.hops_rru1,
                    'hops-rru2'    : ds.hops_rru2,
                    'start-time'   : ds.start_time.strftime('%Y-%m-%d %H:%M:%S')
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


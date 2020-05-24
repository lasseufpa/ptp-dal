from flask_sqlalchemy import SQLAlchemy
import logging


logger = logging.getLogger(__name__)
db     = SQLAlchemy()


class Dataset(db.Model):
    __tablename__ = 'datasets'

    id                 = db.Column(db.Integer, primary_key=True)
    name               = db.Column(db.String(120), unique=True, nullable=False)
    oscillator         = db.Column(db.String(20), nullable=False)
    sync_period        = db.Column(db.Float, nullable=False)
    hops_rru1          = db.Column(db.Integer)
    hops_rru2          = db.Column(db.Integer)
    n_rru_ptp          = db.Column(db.Integer)
    delay_cal          = db.Column(db.Boolean)
    delay_cal_duration = db.Column(db.Integer)
    pipeline_bbu       = db.Column(db.Integer)
    pipeline_rru       = db.Column(db.Integer)
    start_time         = db.Column(db.DateTime)
    departure_ctrl     = db.Column(db.Boolean)
    departure_gap      = db.Column(db.Integer)
    fh_traffic         = db.Column(db.Boolean, nullable=False)
    fh_type            = db.Column(db.String(20))
    fh_fs              = db.Column(db.Float)
    fh_iq_size_dl      = db.Column(db.Integer)
    fh_iq_size_ul      = db.Column(db.Integer)
    fh_bitrate_dl      = db.Column(db.Float)
    fh_bitrate_ul      = db.Column(db.Float)
    fh_n_spf_dl        = db.Column(db.Integer)
    fh_n_spf_ul        = db.Column(db.Integer)
    fh_n_rru_ul        = db.Column(db.Integer)
    fh_n_rru_dl        = db.Column(db.Integer)
    fh_vlan_pcp        = db.Column(db.Integer)

    def __repr__(self):
        return f"Dataset {self.name}"

    def save(self):
        """Save model on the database"""
        db.session.add(self)
        db.session.commit()


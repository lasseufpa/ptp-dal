import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from manager import Manager


class Watcher:
    def __init__(self, app, db):
        """Monitor file system changes

        Args:
            app : Flask app object

        """
        self.ds_folder = "/app/api/datasets/"
        self.manager   = Manager(app, db, self.ds_folder)
        self.handler   = Handler(self.manager)
        self.observer  = Observer()

        # Populate the database on the first run
        self.manager.populate()

    def run(self):
        self.observer.schedule(self.handler, self.ds_folder, recursive=False)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()


class Handler(PatternMatchingEventHandler):
    def __init__(self, manager):
        """Handler constructure"""
        super(Handler, self).__init__(patterns=['*.json', '*.xz'],
                                      ignore_patterns=["*~"],
                                      ignore_directories=True,
                                      case_sensitive=True)
        self.manager = manager
        self.logger  = self.manager.logger

    def on_created(self, event):
        ds_path     = event.src_path
        ds_name     = os.path.splitext(os.path.basename(ds_path))[0]
        ds_metadata = self.manager._read_metadata(ds_path)

        if (ds_metadata):
            self.manager.insert(ds_name, ds_metadata)

    def on_deleted(self, event):
        deleted_ds = os.path.splitext(os.path.basename(event.src_path))[0]
        exists     = self.manager.search_by_name(deleted_ds)
        if (exists):
            self.manager.delete(deleted_ds)

    def on_moved(self, event):
        pass

    def on_modified(self, event):
        pass


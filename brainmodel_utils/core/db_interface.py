import sys

import pymongo
import gridfs
import os
import git
import datetime
import pickle
import time
import subprocess

from functools import wraps
from pymongo.errors import AutoReconnect, DuplicateKeyError, DocumentTooLarge


class DBInterface(object):
    def __init__(self, dbname, port, exp_id, metadata, overwrite=False):
        # These are important for the retry decorator
        self.dbport = int(port)
        self.hostname = "node14-ccncluster.stanford.edu"
        self.max_retries = 10
        self.retry_delay = 0.4
        self.overwrite = overwrite

        self.dbname = dbname
        self._mongo_client = pymongo.MongoClient(f"mongodb://localhost:{self.dbport}/")
        self._database = self._mongo_client[dbname]

        self.check_for_existing_experiment(exp_id, metadata)
        experiments_id = self.save_exp_metadata(exp_id, metadata)

        self.exp_id = exp_id
        self.record_base = {
            "exp_id": exp_id,
            "experiments_id": experiments_id,
        }


    def update_record_base(self, record_base):
        self.record_base.update(record_base)


    def remove_from_record_base(self, key):
        self.record_base.pop(key, None)


    def get_record_base(self):
        # We return a copy of the record_base, such that the ObjectID changes
        return self.record_base.copy()


    def save_per_neuron(self, results, additional_data={}):
        record = self.get_record_base()
        record.update(additional_data)
        for r in results:
            r.update(record)
            self.save_one_neuron(r, is_record=True)


    def retry(func):
        @wraps(func)
        def wrapped_fn(*args, **kwargs):
            num_fails = 0
            max_retries = args[0].max_retries
            dbport = args[0].dbport
            hostname = args[0].hostname

            while True:
                try:
                    return func(*args, **kwargs)
                except AutoReconnect as e:
                    num_fails += 1
                    if num_fails > max_retries:
                        print(f"Reraising exception after {max_retries} retries")
                        raise e
                    # Re-setup the ssh tunnel before retrying
                    delay = args[0].retry_delay*2**num_fails
                    print(f"Retry number {num_fails}, sleeping for {delay} seconds...")
                    time.sleep(delay)
                    subprocess.run([
                        "ssh",
                        "-fNL",
                        f"{dbport}:localhost:{dbport}",
                        f"{hostname}"
                    ])

        return wrapped_fn


    @retry
    def check_for_existing_experiment(self, exp_id, metadata):
        metadata = metadata.copy()
        metadata.pop("overwrite") # We don't want to query on the overwrite flag
        metadata.update({"exp_id": exp_id})
        # Query the DB on this particular run configuration
        # excluding git_hash and run_started
        records = self._database.experiments.find(metadata)
        records = list(records)
        if self.overwrite:
            # Overwrite logic: delete records associated with this experiment
            for record in records:
                self.delete_documents_for_query("experiments", {"_id": record["_id"]})
                for coll in ["per_neuron", "agg_over_neurons", "agg_over_splits", "models"]:
                    self.delete_documents_for_query(coll, {"experiments_id": record["_id"]})
            records = self._database.experiments.find(metadata)
            records = list(records)
        elif len(records) >= 1:
            msg = (
                "An experiment with this configuration exists. "
                "If you with to overwrite, please specify the --overwrite flag \n"
                f"Config: {metadata}"
            )
            raise Exception(msg)
        assert len(records) == 0


    @retry
    def delete_documents_for_query(self, collection_name, query):
        self._database[collection_name].delete_many(query)


    @retry
    def save_exp_metadata(self, exp_id, metadata):
        # Save some info on how/when the experiment was run
        repo_path = os.path.abspath(__file__)
        repo = git.Repo(repo_path, search_parent_directories=True)
        sha = repo.head.object.hexsha
        exp_metadata = {
            "exp_id": exp_id,
            "git_hash": sha,
            "run_started": str(datetime.datetime.now()),
        }
        exp_metadata.update(metadata)

        experiment = self._database.experiments.insert_one(exp_metadata)
        return experiment.inserted_id


    @retry
    def save_one_neuron(self, result, additional_data={}, is_record=False):
        if not is_record:
            record = self.get_record_base()
            record.update(additional_data)
            record.update(result)
        else:
            record = result
        try:
            self._database.per_neuron.insert_one(record)
        except DuplicateKeyError as e:
            # This can only happen if the insert was successful in the first try,
            # but the function failed with an AutoReconnectError once the document
            # was inserted, and was thus retried, so we should ignore it.
            pass


    @retry
    def save_agg_over_neurons(self, result, additional_data={}):
        record = self.get_record_base()
        record.update(additional_data)
        record.update(result)
        self._database.agg_over_neurons.insert_one(record)


    @retry
    def save_agg_over_splits(self, result, additional_data={}):
        record = self.get_record_base()
        record.update(additional_data)
        record.update(result)
        self._database.agg_over_splits.insert_one(record)


    @retry
    def save_agg_over_splithalves(self, result, additional_data={}):
        record = self.get_record_base()
        record.update(additional_data)
        record.update(result)
        self._database.agg_over_splithalves.insert_one(record)


    @retry
    def save_model(self, model, additional_data={}):
        record = self.get_record_base()
        record.update(additional_data)
        gfs = gridfs.GridFS(self._database)
        file_id = gfs.put(pickle.dumps(model))
        record.update({"serialized_model_gfs_id": file_id})
        try:
            self._database.models.insert_one(record)
        except DocumentTooLarge:
            print("Tried to save model but it exceeded 16 MB, so saving failed", file=sys.stderr)

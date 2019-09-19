import logging
import datetime

import decorator
from mock import Mock, patch


from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType, StructField

from core.spark import local_session
from core.utils import MAP_TO_SPARK_TYPE

def as_dicts(dataframe: DataFrame) -> list:
    dataframe = [ row.asDict() for row in dataframe.collect() ]
    return sorted(dataframe, key= lambda row: str(row))

class mock_build_time(object):
    def __init__(self, *time):
        self.mock_time = Mock()
        self.mock_time.return_value = datetime.datetime(*time)

    def __call__(self, test_case):
        def test_wrapper(test_case, *args, **kwargs):
            with patch('core.stages.DataframeTransform.build_time', self.mock_time):
                return test_case(*args, **kwargs)
        return decorator.decorator(test_wrapper, test_case)

class SparkTestCase(object):
    @classmethod
    def setup_class(cls):
        logging.getLogger('py4j').setLevel(logging.ERROR)

    def teardown_method(self, method):
        spark = local_session()
        spark.catalog.clearCache() # cache is shared, we cannot run pytest with multiple threads

        for db in spark.catalog.listDatabases():
            if db.name == 'default':
                continue
            spark.sql('DROP DATABASE {} CASCADE'.format(db.name))

class SparkTemplate(object):

    def __init__(self, schema, table=None):
        self.default_values = schema
        self.spark_schema = StructType()
        self.table = table

        for col, default in schema.items():
            try:
                spark_type = MAP_TO_SPARK_TYPE[type(default)](),
            except KeyError as e:
                raise KeyError("No such spark_type for: '{}' on column '{}'".format(type(default), col))

            self.spark_schema.add(StructField(col, spark_type[0]))

    def __call__(self, data, table=None):
        if table is not None:
            self.table = table

        spark =  local_session()

        data = [ { **self.default_values, **new_values } for new_values in data ]
        data = spark.createDataFrame(data, self.spark_schema)

        if self.table is not None:
            db, table = self.table.split('.')
            spark.sql('CREATE DATABASE IF NOT EXISTS {}'.format(db))
            data.write.saveAsTable(db + '.' + table, mode='append')

        return data

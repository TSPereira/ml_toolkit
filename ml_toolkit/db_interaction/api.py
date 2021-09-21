import warnings
import_err_msg = '\nCould not load {} from {} cause {} is missing from environment. ' \
                 '\nInstall the package in the environment to use this functionality.'

try:
    from .postgresql import PostgreSQLManager
except ModuleNotFoundError as e:
    warnings.warn(import_err_msg.format('PostgreSQLManager', 'ml_toolkit.db_interaction.postgres', e.name),
                  stacklevel=2)

try:
    from .oracle import OracleManager
except ModuleNotFoundError as e:
    warnings.warn(import_err_msg.format('OracleManager', 'ml_toolkit.db_interaction.postgres', e.name), stacklevel=2)


try:
    from .mongodb import MongoManager
except ModuleNotFoundError as e:
    warnings.warn(import_err_msg.format('MongoManager', 'ml_toolkit.db_interaction.mongodb', e.name), stacklevel=2)
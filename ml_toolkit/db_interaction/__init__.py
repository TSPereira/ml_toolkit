import_error_msg = 'Could not load {} from {} cause {} is missing from environment. ' \
                   'Install the package in the environment to use this functionality'

try:
    from .postgres import PostgreSQLManager
except ModuleNotFoundError as e:
    print(import_error_msg.format('PostgreSQLManager', 'support_toolkit.ml_toolkit.db_interaction.postgres', e.name))

try:
    from .oracle import OracleManager
except ModuleNotFoundError as e:
    print(import_error_msg.format('OracleManager', 'support_toolkit.ml_toolkit.db_interaction.postgres', e.name))

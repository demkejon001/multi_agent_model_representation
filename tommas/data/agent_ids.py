import os

agent_id_dirpath = 'data/agent_models/'
agent_id_filepath = agent_id_dirpath + 'agent_ids.txt'


def check_if_file_exists():
    return os.path.isfile(agent_id_filepath)


def create_new_file():
    if not os.path.exists(agent_id_dirpath):
        os.makedirs(agent_id_dirpath)
    file = open(agent_id_filepath, "w+")
    file.write('-2\n')
    file.write('-1')
    file.close()


def append_agent_ids(agent_ids):
    file = open(agent_id_filepath, "a")
    for agent_id in agent_ids:
        file.write("\n%d" % (agent_id,))
    file.close()


def get_agent_ids(num_agent_ids=1):
    if not check_if_file_exists():
        create_new_file()
    last_recorded_agent_id = get_last_recorded_agent_id()
    start_agent_id = last_recorded_agent_id + 1
    new_agent_ids = []
    for _ in range(num_agent_ids):
        new_agent_ids.append(start_agent_id)
        start_agent_id += 1
    append_agent_ids(new_agent_ids)
    return new_agent_ids


def get_last_recorded_agent_id():
    with open(agent_id_filepath, 'rb') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()
    return int(last_line)


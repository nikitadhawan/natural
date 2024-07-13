import zstandard
import os
import json
import pandas as pd
import wget
import logging.handlers

def download_pushshift(subreddit, data_type, data_path):
    _= wget.download( 'https://the-eye.eu/redarcs/files/{}_{}.zst'.format(subreddit, data_type), out=data_path)

    log = logging.getLogger("bot")
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler())

    def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
        chunk = reader.read(chunk_size)
        bytes_read += chunk_size
        if previous_chunk is not None:
            chunk = previous_chunk + chunk
        try:
            return chunk.decode()
        except UnicodeDecodeError:
            if bytes_read > max_window_size:
                raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
            log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
            return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


    def read_lines_zst(file_name):
        with open(file_name, 'rb') as file_handle:
            buffer = ''
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
            while True:
                chunk = read_and_decode(reader, 2**27, (2**29) * 2)
                if not chunk:
                    break
                lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                    yield line, file_handle.tell()
            buffer = lines[-1]
        reader.close()


    file_path = data_path + '{}_{}.zst'.format(subreddit, data_type)
    file_size = os.stat(file_path).st_size
    file_lines = 0
    file_bytes_processed = 0
    created = None
    bad_lines = 0
    data = []

    for line, file_bytes_processed in read_lines_zst(file_path):
        try:
            obj = json.loads(line)
            data += [obj]
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
        file_lines += 1
        if file_lines % 100000 == 0:
            log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

    save_path = data_path + '{}_{}.csv'.format(subreddit, data_type)
    data_csv = pd.DataFrame(data)
    data_csv.to_csv(save_path)
    os.remove(file_path)
    log.info(f" Complete : {file_lines:,} : {bad_lines:,}")
    
import os
import random
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import os
from tqdm import tqdm


def process_file(file):
    if file.endswith(".wav"):
        file = file[:-4]
        path_spk = f"./data_svc/speaker/{spks}/{file}.spk.npy"
        path_wave = f"./data_svc/waves-32k/{spks}/{file}.wav"
        path_spec = f"./data_svc/specs/{spks}/{file}.pt"
        path_pitch = f"./data_svc/pitch/{spks}/{file}.pit.npy"
        path_whisper = f"./data_svc/whisper/{spks}/{file}.ppg.npy"
        assert os.path.isfile(path_spk), path_spk
        assert os.path.isfile(path_wave), path_wave
        assert os.path.isfile(path_spec), path_spec
        assert os.path.isfile(path_pitch), path_pitch
        assert os.path.isfile(path_whisper), path_whisper
        all_items.append(
            f"{path_wave}|{path_spec}|{path_pitch}|{path_whisper}|{path_spk}")
    return all_items

if __name__ == "__main__":
    os.makedirs("./files/", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    args = parser.parse_args()
    
    rootPath = "./data_svc/waves-32k/"
    all_items = []
    for spks in os.listdir(f"./{rootPath}"):
        if os.path.isdir(f"./{rootPath}/{spks}"):
            if args.thread_count == 0:
                process_num = os.cpu_count()
            else:
                process_num = args.thread_count
            with ThreadPoolExecutor(max_workers=process_num) as executor:
                for spks in os.listdir(f"./{rootPath}"):
                    futures = [executor.submit(process_file, file) for file in os.listdir(f"./{rootPath}/{spks}")]
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        pass

    random.shuffle(all_items)
    valids = all_items[:10]
    valids.sort()
    trains = all_items[10:]
    # trains.sort()
    fw = open("./files/valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open("./files/train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
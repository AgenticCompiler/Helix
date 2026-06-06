import json

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=2)

def write_txt(path, data):
    with open(path, "w", encoding="utf-8", errors="replace") as out_f:
        out_f.writelines(data)
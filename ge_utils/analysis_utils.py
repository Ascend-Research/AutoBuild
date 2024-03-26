def report_categorical(kv_dict):

    kv_list = [[k, v] for k, v in kv_dict.items()]
    kv_list.sort(reverse=True, key=lambda x: x[-1])

    for i, kv in enumerate(kv_list):
        print(f" - Rank {i}: {kv[0]} ({kv[1]})")

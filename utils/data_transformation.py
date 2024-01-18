def get_id_2_label_dict(data_text, label_name, dataset_name):
    '''
    {text_input_id: label}
    '''
    labels = {}
    for item in data_text:
        text_input_id = item["text_input_id"]
        label_value = item[label_name]
        labels[text_input_id] = label_value

    return labels
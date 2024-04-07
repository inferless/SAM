INPUT_SCHEMA = {
    "image_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"]
    },
    "input_points": {
        'datatype': 'FP32',  # Assuming coordinates are integers
        'required': True,
        'shape': [2],  # Specifies that exactly two integers are expected
        'example': [[1740,748]]  # Provides an example as a nested list, consistent with your shape requirement
    }
    
}

INPUT_SCHEMA = {
    "image_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://www.godrejinterio.com/imagestore/B2C/56101543SD00165/56101543SD00165_A2_803x602.jpg"]
    },
    "input_points": {
        'datatype': 'FP32',  # Assuming coordinates are integers
        'required': True,
        'shape': [2],  # Specifies that exactly two integers are expected
        'example': [[1740,748]]  # Provides an example as a nested list, consistent with your shape requirement
    }
    
}

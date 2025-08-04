import ctypes
import numpy as np


class TFLiteCModel:
    def __init__(self, model_path, lib_path):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.lib.TFL_LoadModel.restype = ctypes.c_void_p
        self.model = self.lib.TFL_LoadModel(model_path.encode('utf-8'))

    def predict(self, input_tensor: np.ndarray):
        assert input_tensor.dtype == np.float32
        input_ptr = input_tensor.ctypes.data_as(ctypes.c_void_p)
        result_ptr = self.lib.TFL_Predict(self.model, input_ptr)
        result_array = np.ctypeslib.as_array(
            (ctypes.c_float * 51).from_address(result_ptr)
        )
        return result_array.reshape(1, 1, 17, 3)

    def __del__(self):
        self.lib.TFL_FreeModel(self.model)

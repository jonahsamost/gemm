import cutlass
import cutlass.cute as cute
from cutelass.base_dsl.typing import Integer
from cutlass.cutlass_dsl import dsl_user_op


class FastDivmod(cute.FastDivmodDivisor):
    @dsl_user_op
    def __init__(
        self,
        divisor: Integer,
        is_power_of_2: bool = None,
        *, loc=None, ip=None
    ):
        super().__init__(divisor, is_power_of_2=is_power_of_2, loc=loc, ip=ip)
        self.divisor = divisor
    
    def __extract_mlir_values__(self):
        # host side 
        return [self._divisor] + cutlass.extract_mlir_values(self.divisor)
    
    def __new_from_mlir_values__(self, values):
        # reconstruct on device side from MLIR values
        new_obj = object.__new__(FastDivmod)
        new_obj._divisor = values[0]
        new_obj.divisor = cutlass.new_from_mlir_values(self.divisor, values[1:])
        return new_obj

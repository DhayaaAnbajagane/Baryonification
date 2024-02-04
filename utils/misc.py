import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, abs, neg, pos


def generate_operator_method(op):
    '''
    Define a method for generating the simple arithmetic
    operations for the Profile classes. This changes the _real()
    routine in the ``HaloProfile'' classes such that the new result
    is the same as generating the _real() of the individual classes
    and then performing the arithmetic operation on them.
    '''

    if op in [add, mul, sub, pow, truediv]:
        def operator_method(self, other):

            assert isinstance(other, (int, float, ccl.halos.profiles.HaloProfile)), f"Object must be int/float/SchneiderProfile but is type '{type(other).__name__}'."


            Combined = self.__class__(**self.model_params, xi_mm = self.xi_mm, R_range = self.R_range)

            def __tmp_real__(cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

                A = self._real(cosmo, r, M, a, mass_def)

                if isinstance(other, ccl.halos.profiles.HaloProfile):
                    B = other._real(cosmo, r, M, a, mass_def)
                else:
                    B = other

                return op(A, B)

            def __str_prf__():

                op_name = op.__name__
                cl_name = self.__str_prf__()
                ot_name = other.__class__.__name__ if isinstance(other, ccl.halos.profiles.HaloProfile) else other

                return f"{op_name}[{cl_name}, {ot_name}]"

            Combined._real = __tmp_real__
            Combined.__str_prf__ = __str_prf__

            return Combined

    #For some operators we don't need a second input, so rewrite function for that
    elif op in [abs, neg, pos]:

        def operator_method(self):
            
            Base     = self.__class__.__bases__[0] #Get the base class of the profile (Normally, SchneiderProfiles)
            Combined = self.__class__(**self.model_params, xi_mm = self.xi_mm, R_range = self.R_range)

            def __tmp_real__(cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

                A = self._real(cosmo, r, M, a, mass_def)

                return op(A)

            def __str_prf__():

                op_name = op.__name__
                cl_name = self.__str_prf__()

                return f"{op_name}[{cl_name}]"

            Combined._real = __tmp_real__
            Combined.__str_prf__ = __str_prf__

            return Combined

    return operator_method
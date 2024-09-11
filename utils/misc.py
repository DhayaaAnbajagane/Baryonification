import numpy as np
import pyccl as ccl
from operator import add, mul, sub, truediv, pow, abs, neg, pos

__all__ = ['generate_operator_method']

def generate_operator_method(op, reflect = False):
    """
    Defines a method for generating simple arithmetic operations for the Profile classes.

    The `generate_operator_method` function dynamically creates methods that can be used to perform
    arithmetic operations (such as addition, subtraction, multiplication, etc.) on instances of the
    `HaloProfile` classes or similar. This function alters the `_real()` routine in the `HaloProfile`
    classes so that the new result is equivalent to computing the `_real()` of the individual classes
    and then performing the specified arithmetic operation.

    Parameters
    ----------
    op : function
        The arithmetic operation to be applied. It should be one of the Python arithmetic operators like
        `add`, `mul`, `sub`, `pow`, `truediv`, `abs`, `neg`, or `pos` from the `operator` module.
    
    reflect : bool, optional
        If `True`, the order of the operands is reversed (i.e., the operation is performed with the
        second operand as the first). Used to handle right-handed/left-handed operations. Default is `False`.

    Returns
    -------
    operator_method : function
        A function that defines how the arithmetic operation should be performed on the `HaloProfile`
        classes. This function can handle operations with other `HaloProfile` instances, integers, or
        floats.

    Examples
    --------
    To use `generate_operator_method` for addition, you might do:

    >>> from operator import add
    >>> add_method = generate_operator_method(add)
    >>> profile_sum = add_method(profile1, profile2)
    
    Notes
    -----
    - The method returned can handle both unary and binary operations depending on the operator specified.
    """

    if op in [add, mul, sub, pow, truediv]:
        def operator_method(self, other):

            assert isinstance(other, (int, float, ccl.halos.profiles.HaloProfile)), f"Object must be int/float/SchneiderProfile but is type '{type(other).__name__}'."


            Combined = self.__class__(**self.model_params, xi_mm = self.xi_mm, 
                                      padding_lo_proj = self.padding_lo_proj, 
                                      padding_hi_proj = self.padding_hi_proj, 
                                      n_per_decade_proj = self.n_per_decade_proj,)

            def __tmp_real__(cosmo, r, M, a, mass_def = ccl.halos.massdef.MassDef(200, 'critical')):

                A = self._real(cosmo, r, M, a, mass_def)

                if isinstance(other, ccl.halos.profiles.HaloProfile):
                    B = other._real(cosmo, r, M, a, mass_def)
                else:
                    B = other

                if not reflect:
                    return op(A, B)
                else:
                    return op(B, A)

            def __str_prf__():

                op_name = op.__name__
                cl_name = self.__str_prf__()
                
                #Check if it has cutom string output already
                #If not then use the class name. Or if int/float just use number
                if isinstance(other, ccl.halos.profiles.HaloProfile):
                    if hasattr(other, '__str_prf__'):
                        ot_name = other.__str_prf__()
                    else:
                        ot_name = other.__class__.__name__ 
                else: 
                    ot_name = other

                
                if not reflect:
                    return f"{op_name}[{cl_name}, {ot_name}]"
                else:
                    return f"{op_name}[{ot_name}, {cl_name}]"

            Combined._real = __tmp_real__
            Combined.__str_prf__ = __str_prf__

            return Combined

    #For some operators we don't need a second input, so rewrite function for that
    elif op in [abs, neg, pos]:

        def operator_method(self):
            
            Base     = self.__class__.__bases__[0] #Get the base class of the profile (Normally, SchneiderProfiles)
            Combined = self.__class__(**self.model_params, xi_mm = self.xi_mm, 
                                      padding_lo_proj = self.padding_lo_proj, 
                                      padding_hi_proj = self.padding_hi_proj, 
                                      n_per_decade_proj = self.n_per_decade_proj)

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


def destory_Pk(cosmo):
    """
    Removes some SwigPyObject from cosmo that are unable to be pickled
    and therefore caused the multiprocessing to break.

    Parameters
    ----------
    cosmo : object
        A ccl Cosmology object
    
    Returns
    -------
    cosmo : object
        The ccl Cosmology object but with some attributes (that are necessarily
        SwigPyObjects) destoryed in order to allow pickling.

    Notes
    -----
    - For efficiency in pipeline, this function should be used on a Cosmology object
    only once the main profile/cosmology operations are finished. Otherwise this
    forces the recalculation of the power spectrum (which may not be an issue depending
    on your use-case, but something to be aware of).
    """

    cosmo._pk_lin = {}
    cosmo._pk_nl  = {}

    return cosmo
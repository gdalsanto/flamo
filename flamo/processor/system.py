import torch
import torch.nn as nn
import warnings
from collections import OrderedDict
from flamo.utils import to_complex


# ============================= SERIES ================================


class Series(nn.Sequential):        
    r"""
    Module for cascading multiple DSP modules in series. Inherits from :class:`nn.Sequential`.
    This class serves as a container for a series of DSP modules, allowing them 
    to be cascaded in a single module. It ensures that all included modules 
    share the same values for the `nfft` and `gamma` attributes, hence all parsed 
    modules are expected to have these attributes.

        **Args**:
            - *args: A variable number of DSP modules of the type :class:`nn.Module`, :class:`nn.Sequential`, or :class:`OrderedDict`.
    """
    def __init__(self, *args):
        super().__init__(self.__unpack_modules(modules=args, current_keys=[]))
        
        # Check nfft and alpha values
        self.nfft = self.__check_attribute('nfft')
        self.gamma = self.__check_attribute('gamma')

    def __unpack_modules(self, modules: tuple, current_keys: list) -> OrderedDict:
        r"""
        Generate an :class:`OrderedDict` containing the modules given in the input 
        tuple, and give a key to each module.
        If any module in the input tuple is a :class:`nn.Sequential` (or :class:`OrderedDict`), 
        it is recognized as nested :class:`nn.Sequential` (or :class:`OrderedDict`),
        it is unpacked, and its modules are added to the output :class:`OrderedDict`.
        For each encountered module, key generation follows these rules:
            1. if the analyzed module is :class:`nn.Sequential` or :class:`OrderedDict`, it is unpacked, thus, if it was nested inside
                another :class:`nn.Sequential` or :class:`OrderedDict`, the key given to itself is neglected. This until the analyzed module is
                a `leaf` module (i.e. :class:`nn.Module` and not :class:`nn.Sequential` or :class:`OrderedDict`), then rules 2-5 are applied
            2. if the module has a unique custom key (e.g. 'my_module'), it is used as is
            3. if the module has a custom key, but such key was previously used for another module (hence it is not unique). An error is raised
            4. if the module has no key, a key is generated for it, equal to its position in the series. A warning is raised
            5. if a custom key is found and can be converted to an integer (e.g. '3'), such key is considered missing
                * rule 4 is applied

        **Args**:
            modules (tuple): The input modules.
            current_keys (list): The current keys of the already unpacked modules.

        **Returns**:
            The unpacked modules (:class:`OrderedDict`).

        **Raises**:
            ValueError: If modules are not of type :class:`nn.Module`, :class:`nn.Sequential`, or :class:`OrderedDict`.
            ValueError: If a custom key is not unique.
        """
        # initialize the unpacked modules as empty OrderedDict
        unpacked_modules = OrderedDict()
        # iterate over the modules
        for module in modules:
            if isinstance(module, nn.Sequential):
                # module is a nn.Sequential
                unpacked_modules.update(
                    self.__unpack_modules(
                        (module._modules,), [*current_keys, *unpacked_modules.keys()]
                    )
                )
            elif isinstance(module, OrderedDict):
                for k, v in module.items():
                    if isinstance(v, nn.Sequential):
                        # nested nn.Sequential
                        unpacked_modules.update(
                            self.__unpack_modules(
                                (v._modules,), [*current_keys, *unpacked_modules.keys()]
                            )
                        )
                    elif isinstance(v, OrderedDict):
                        # nested OrderedDict
                        unpacked_modules.update(
                            self.__unpack_modules(
                                (v,), [*current_keys, *unpacked_modules.keys()]
                            )
                        )
                    else:
                        try:
                            int(k)
                            # custom key is convertible to integer, key is overwritten
                            new_key = str(len(unpacked_modules) + len(current_keys))
                            unpacked_modules[new_key] = v
                            if k != new_key:
                                warnings.warn(f"Key {k} is an integer, it will be overwritten.")
                        except: # custom key found
                            if k in current_keys or k in unpacked_modules.keys():
                                # custom key is not unique
                                raise ValueError(f"Key {k} is already present in the Series.")
                            # custome key is unique
                            unpacked_modules[k] = v
            elif isinstance(module, nn.Module):
                # module without key
                unpacked_modules[str(len(unpacked_modules) + len(current_keys))] = (
                    module
                )
            else:
                raise ValueError("Modules must be nn.Module, nn.Sequential, or OrderedDict.")
            
        return unpacked_modules
    
    def __check_attribute(self, attr: str) -> int | float | None:
        r"""
        Checks if all modules have the same value of the requested attribute.

            **Args**:
                - attr (str): The attribute to check.

            **Returns**:
                - int | float | None: The attribute value.

            **Raises**:
                ValueError: If at least one module with incorrect attribute value was found.
        """
        value = None
        # First, store the value of the attribute if found in any of modules.
        for module in self:
            if hasattr(module, attr):
                value = getattr(module, attr)
                break
        if value is None:
            warnings.warn(f"Attribute {attr} not found in any of the modules.")
        # Then, check if all modules have the same value of the attribute. If any module has a different value, raise an error.
        for i,module in enumerate(self):
            if hasattr(module, attr):
                if getattr(module, attr) != value:
                    raise ValueError(f"All modules must have the same {attr} value. Module at index {i} is incoherent with the part of the Series preceding it.")
        
        return value


# ============================= RECURSION ================================


class Recursion(nn.Module):
    r"""
    Recursion module for computing closed-loop transfer function. Inherits from :class:`nn.Module`.
    The feedforward and feedback paths if are given as a :class:`nn.Module`, :class:`nn.Sequential`, or :class:`OrderedDict`,
    they are converted to a :class:`Series` instance. 

        **Args**:
            - Ff: The feedforward path with size (nfft, m, n).
            - Fb: The feedback path with size (nfft, n, m).
            - alpha (float, optional): The alpha value. Defaults to None.

    For details on the closed-loop transfer function: <https://en.wikipedia.org/wiki/Closed-loop_transfer_function>`_.
    """
    def __init__(self,
                 fF: nn.Module | nn.Sequential | OrderedDict | Series,
                 fB: nn.Module | nn.Sequential | OrderedDict | Series,
                ):
        # Prepare the feedforward and feedback paths
        if isinstance(fF, (nn.Sequential, OrderedDict)) and not isinstance(fF, Series):
            fF = Series(fF)
            warnings.warn('Feedforward path has been converted to a Series class instance.')
        if isinstance(fB, (nn.Sequential, OrderedDict)) and not isinstance(fB, Series):
            fB = Series(fB)
            warnings.warn('Feedback path has been converted to a Series class instance.')

        super().__init__()
        self.feedforward = fF
        self.feedback = fB

        # Check nfft and alpha values
        self.nfft = self.__check_attribute('nfft')
        # self.alpha = self.__check_attribute('gamma')

        # Check I/O compatibility
        # self.input_channels, self.output_channels = self.__check_io()


    def forward(self, X):
        """
        Forward pass of the Recursion module.

        Args:
            - X (torch.Tensor): The input tensor with shape (batch, nfft, n, ..).

        Returns:
            - torch.Tensor: The output tensor with shape (batch, nfft, m, ..).
        """
        B = self.feedforward(X)
       
        # Identity matrix
        if not hasattr(self, "I"):
            self.I = torch.zeros(
                B.shape[1], B.shape[-1], B.shape[-1], dtype=torch.complex64
            )
            for i in range(B.shape[-1]):
                self.I[:, i, i] = 1

        # expand identity matrix to batch size
        expand_dim = [X.shape[0]] + [d for d in self.I.shape]
        I = self.I.expand(tuple(expand_dim))

        HH = self.feedback(I)
        A = I - self.feedforward(HH)
        return torch.linalg.solve(A, B)
    
    # ---------------------- Check methods ----------------------
    def __check_attribute(self, attr: str) -> int | float:
        """
        Checks if feedforward and feedback paths have the same value of the requested attribute

        Args:
            - attr (str): The attribute to check.

        Returns:
            - int | float: The attribute value.

        Raises:
            ValueError: The two paths have different values of the requested attribute.
        """
        # First, check that both feedforward and feedback paths possess the attribute.
        if getattr(self.feedforward, attr, None) is None:
            raise ValueError(f"The feedforward pass does not possess the attribute {attr}.")
        if getattr(self.feedback, attr, None) is None:
            raise ValueError(f"The feedback pass does not possess the attribute {attr}.")
        # Then, check that the two paths have the same value of the attribute.
        assert getattr(self.feedforward, attr) == getattr(self.feedback, attr), f"Feedforward and feedback paths must have the same {attr} value."

        return getattr(self.feedforward, attr)
    
    def __check_io(self) -> tuple:
        # NOTE: still work in progress
        # NOTE: does not work for SVF
        """
        Check if the feedforward and feedback paths have compatible input/output shapes.
        """
        # Get input channels of both feedforward and feedback
        if isinstance(self.feedforward, Series):
            ff_in_ch = self.feedforward[0].size[-1]
        else:
            ff_in_ch = self.feedforward.size[-1]
        if isinstance(self.feedback, Series):
            fb_in_ch = self.feedback[0].size[-1]
        else:
            fb_in_ch = self.feedback.size[-1]

        # Found out the number of output channels of both feedforward and feedback
        x = to_complex(torch.zeros(1, self.nfft//2+1, ff_in_ch))
        ff_out_ch = self.feedforward(x).shape[-1]
        x = to_complex(torch.zeros(1, self.nfft//2+1, fb_in_ch))
        fb_out_ch = self.feedback(x).shape[-1]

        # Check if the input/output channels are compatible
        assert(ff_out_ch == fb_in_ch), "Feedforward output channels and feedback input channels must have the same."
        assert(fb_out_ch == ff_in_ch), "Feedforward input channels and feedback output channels must have the same."

        return ff_in_ch, ff_out_ch

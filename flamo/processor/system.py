import torch
import torch.nn as nn
import warnings
from collections import OrderedDict
from flamo.processor.dsp import FFT, iFFT, Transform
from flamo.functional import signal_gallery

# ============================= SERIES ================================


class Series(nn.Sequential):
    r"""
    Module for cascading multiple DSP modules in series. Inherits from :class:`nn.Sequential`.
    This class serves as a container for a series of DSP modules (preferably constructed from :class:`flamo.processor.dsp.DSP`), allowing them
    to be cascaded in a single module. It ensures that all included modules
    share the same values of :attr:`nfft` and :attr:`alias_decay_db` attributes, hence all parsed
    modules are expected to have these attributes.

        **Arguments**:
            **\*args**: An arbitrary number of DSP modules of the type :class:`nn.Module`, :class:`nn.Sequential`, or :class:`OrderedDict`.
    """

    def __init__(self, *args):
        super().__init__(self.__unpack_modules(modules=args, current_keys=[]))

        # Check nfft and alpha values
        self.nfft = self.__check_attribute("nfft")
        self.alias_decay_db = self.__check_attribute("alias_decay_db")

        # Check I/O compatibility
        self.input_channels, self.output_channels = self.__check_io()

    def prepend(self, new_module: nn.Module | nn.Sequential | OrderedDict) -> "Series":
        r"""
        Prepends a given item to the beginning of the :class:`Series` instance.

            **Arguments**:
                **new_module** (nn.Module | nn.Sequential | OrderedDict): item to append.

            **Returns**:
                Series: self.
        """
        return self.insert(index=0, new_module=new_module)

    def append(self, new_module: nn.Module | nn.Sequential | OrderedDict) -> "Series":
        r"""
        Appends a given item to the end of the :class:`Series` instance.

            **Arguments**:
                **new_module** (nn.Module | nn.Sequential | OrderedDict): item to append.

            **Returns**:
                Series: self.
        """
        # Get current keys
        current_keys = self._modules.keys()

        # Unpack the new module
        unpacked_modules = self.__unpack_modules((new_module,), [*current_keys])

        # Add the unpacked modules at the end of the Series
        for k, v in unpacked_modules.items():
            self.add_module(k, v)

        # Check nfft and alpha values
        self.nfft = self.__check_attribute("nfft")
        self.alias_decay_db = self.__check_attribute("alias_decay_db")

        # Check I/O compatibility
        self.input_channels, self.output_channels = self.__check_io()

        return self

    def insert(
        self, index: int, new_module: nn.Module | nn.Sequential | OrderedDict
    ) -> "Series":
        r"""
        Inserts a given item at the given index in the Series instance.

            **Arguments**:
                - **index** (int): index at which to insert the new module.
                - **new_module** (nn.Module | nn.Sequential | OrderedDict): item to append.

            **Returns**:
                Series: self.
        """
        # Check that the index is within the range of the current modules
        n_current_modules = len(self._modules)
        if not (-n_current_modules <= index <= n_current_modules):
            raise IndexError(f"Index out of range.")
        if index < 0:
            index += n_current_modules

        # Get current keys
        current_keys = self._modules.keys()

        # Unpack the new module
        unpacked_modules = list(
            self.__unpack_modules((new_module,), [*current_keys]).items()
        )

        # Get current keys-modules
        current_items = self._modules

        # Replicate the current modules in list
        items = list(current_items.items())

        # Add new modules to the list
        for i in range(index, index + len(unpacked_modules)):
            items.insert(i, unpacked_modules[i - index])

        # Propagate addition to original modules
        current_items.clear()
        current_items.update(items)

        # Check nfft and alpha values
        self.nfft = self.__check_attribute("nfft")
        self.alias_decay_db = self.__check_attribute("alias_decay_db")

        # Check I/O compatibility
        self.input_channels, self.output_channels = self.__check_io()

        return self

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
            5. if a custom key is found and can be converted to an integer (e.g. '3'), such key is considered missing and rule 4 is applied

        **Arguments**:
            - **modules** (tuple): The input modules.
            - **current_keys** (list): The current keys of the already unpacked modules.

        **Returns**:
            The unpacked modules (:class:`OrderedDict`).

        **Raises**:
            - ValueError: If modules are not of type :class:`nn.Module`, :class:`nn.Sequential`, or :class:`OrderedDict`.
            - ValueError: If a custom key is not unique.
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
            elif isinstance(module, OrderedDict) or isinstance(module, dict):
                for k, v in module.items():
                    if isinstance(v, nn.Sequential):
                        # nested nn.Sequential
                        unpacked_modules.update(
                            self.__unpack_modules(
                                (v._modules,), [*current_keys, *unpacked_modules.keys()]
                            )
                        )
                    elif isinstance(v, OrderedDict) or isinstance(v, dict):
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
                                warnings.warn(
                                    f"Key {k} is an integer, it will be overwritten."
                                )
                        except:  # custom key found
                            if k in current_keys or k in unpacked_modules.keys():
                                # custom key is not unique
                                raise ValueError(
                                    f"Key {k} is already present in the Series."
                                )
                            # custome key is unique
                            unpacked_modules[k] = v
            elif isinstance(module, nn.Module):
                # module without key
                unpacked_modules[str(len(unpacked_modules) + len(current_keys))] = (
                    module
                )
            else:
                raise ValueError(
                    "Modules must be nn.Module, nn.Sequential, or OrderedDict."
                )

        return unpacked_modules

    def __check_attribute(self, attr: str) -> int | float | None:
        r"""
        Checks if all modules have the same value of the requested attribute.

            **Arguments**:
                **attr** (str): The attribute to check.

            **Returns**:
                int | float | None: The attribute value.

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
        else:
            for i, module in enumerate(self):
                if hasattr(module, attr) and getattr(module, attr) != value:
                    raise ValueError(
                        f"All modules must have the same {attr} value. Module {module.__class__.__name__} at index {i} is incoherent with the part of the Series preceding it."
                    )

        return value

    def __check_io(self):
        r"""
        Checks if the modules in the Series have compatible input/output shapes.

            **Returns**:
                tuple(int,int): The number of input and output channels.
        """
        found = False
        input_channels = None
        prev_out_channels = None

        for i, module in enumerate(self):
            try:
                input_channels = getattr(module, "input_channels")
                found = True
                break
            except:
                continue

        if found:
            prev_module = self[i].__class__.__name__
            prev_position = i
            prev_out_channels = self[i].output_channels

            for j, module in enumerate(self):
                if j <= i:
                    continue
                if hasattr(module, "input_channels"):
                    assert (
                        getattr(module, "input_channels") == prev_out_channels
                    ), f"Module {prev_module} at index {prev_position} has {prev_out_channels} output channels, but module {module.__class__.__name__} at index {j} has {module.input_channels} input_channels."
                    prev_module = module.__class__.__name__
                    prev_position = j
                    prev_out_channels = getattr(module, "output_channels", None)

        return input_channels, prev_out_channels

    def forward(self, input, ext_param=None):
        r"""
        Forward pass through the Series.

                **Arguments**:
                    - **input** (Tensor): The input tensor.
                    - **ext_param** (torch.Tensor, optional): Parameter values received from external modules (hyper conditioning). Default: None.

                **Returns**:
                    Tensor: The output tensor.
        """

        if ext_param is not None:
            for key, module in self._modules.items():
                # check if the key is in param_dict
                if ext_param is not None and key in ext_param:
                    input = module(input, ext_param[key])
                else:
                    input = module(input)
        else:
            for module in self:
                input = module(input)
        return input


# ============================= RECURSION ================================


class Recursion(nn.Module):
    r"""
    Recursion module for computing closed-loop transfer function. Inherits from :class:`nn.Module`.
    The feedforward and feedback paths if are given as a :class:`nn.Module`, :class:`nn.Sequential`, or :class:`OrderedDict`,
    they are converted to a :class:`Series` instance.

    Shape:
        - input: :math:`(B, M, N_{in}, ...)`
        - output: :math:`(B, M, N_{out}, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    :math:`N_{in}` is the number of input channels, and :math:`N_{out}` is the number of output channels.
    Ellipsis :math:`(...)` represents additional dimensions.

        **Arguments**:
            - **fF**: The feedforward path with size (M, N_{out}, N_{in}).
            - **fB**: The feedback path with size (M, N_{in}, N_{out}).
            - **alias_decay_db** (float, optional): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Defaults to None.

        **Attributes**:
            - **feedforward** (nn.Module | Series): The feedforward path.
            - **feedback** (nn.Module | Series): The feedback path.
            - **nfft** (int): The number of frequency points.
            - **alias_decay_db** (float): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples.

    For details on the closed-loop transfer function see `Wikipedia page <https://en.wikipedia.org/wiki/Closed-loop_transfer_function>`_.
    """

    def __init__(
        self,
        fF: nn.Module | nn.Sequential | OrderedDict | Series,
        fB: nn.Module | nn.Sequential | OrderedDict | Series,
    ):

        nn.Module.__init__(self)

        # Prepare the feedforward and feedback paths
        if isinstance(fF, (nn.Sequential, OrderedDict)) and not isinstance(fF, Series):
            self.feedforward = Series(fF)
            warnings.warn(
                "Feedforward path has been converted to a Series class instance."
            )
        else:
            self.feedforward = fF
        if isinstance(fB, (nn.Sequential, OrderedDict)) and not isinstance(fB, Series):
            self.feedback = Series(fB)
            warnings.warn(
                "Feedback path has been converted to a Series class instance."
            )
        else:
            self.feedback = fB

        # Check nfft and time anti-aliasing decay-envelope parameter values
        self.nfft = self.__check_attribute("nfft")
        self.alias_decay_db = self.__check_attribute("alias_decay_db")

        # Check I/O compatibility
        self.input_channels, self.output_channels = self.__check_io()

        # Identity matrix for the forward computation
        self.I = self.__generate_identity().to(device=self.alias_decay_db.device)

    def forward(self,  X: torch.Tensor, ext_param: dict = None):
        r"""
        Applies the closed-loop transfer function to the input tensor X.

            **Arguments**:
                **X** (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.

            **Returns**:
                torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        ext_param_fb = None
        ext_param_ff = None
        if ext_param is not None:
            for key, param in ext_param.items():
                # check if the key is in param_dict
                if 'feedback' in key:
                    ext_param_fb = param
                elif 'feedforward' in key:
                    ext_param_ff = param

        B = self.feedforward(X, ext_param_ff)

        # Expand identity matrix to batch size
        expand_dim = [X.shape[0]] + [d for d in self.I.shape]
        I = self.I.expand(tuple(expand_dim))

        HH = self.feedback(I, ext_param_fb)
        A = I - self.feedforward(HH, ext_param_ff)
        return torch.linalg.solve(A, B)

    def __generate_identity(self) -> torch.Tensor:
        r"""
        Generates the identity matrix necessary for the forward computation.

            **Returns**:
                torch.Tensor: The identity matrix.
        """
        size = (self.nfft // 2 + 1, self.output_channels, self.output_channels)
        I = torch.complex(torch.zeros(*size), torch.zeros(*size))
        for i in range(self.output_channels):
            I[:, i, i] = 1
        return I

    # ---------------------- Check methods ----------------------
    def __check_attribute(self, attr: str) -> int | float:
        r"""
        Checks if feedforward and feedback paths have the same value of the requested attribute.

            **Arguments**:
                **attr** (str): The attribute to check.

            **Returns**:
                int | float: The attribute value.

            **Raises**:
                ValueError: The two paths have different values of the requested attribute.
        """
        ff_attr = getattr(self.feedforward, attr, None)
        if ff_attr is None:
            warnings.warn(
                f"The feedforward pass does not possess the attribute {attr}."
            )
        fb_attr = getattr(self.feedback, attr, None)
        if fb_attr is None:
            warnings.warn(f"The feedback pass does not possess the attribute {attr}.")
        # Then, check that the two paths have the same value of the attribute.
        if ff_attr is not None and fb_attr is not None:
            assert (
                ff_attr == fb_attr
            ), f"The feedforward pass has {attr} = {ff_attr} and feedback pass has {attr} = {fb_attr}. They must have the same value."

        # Return the attribute value
        attr_value = ff_attr if ff_attr is not None else fb_attr

        return attr_value

    def __check_io(self) -> tuple:
        r"""
        Checks if the feedforward and feedback paths have compatible input/output shapes.

            **Returns**:
                tuple(int,int): The number of input and output channels.

            **Raises**:
                - ValueError: The feedforward or the feedback paths do not possess either the input_channels or the output_channels attributes.
                - AssertionError: The feedforward and the feedback paths' input and output channels are not compatible.
        """
        # Get input channels of both feedforward and feedback
        ff_in_ch = getattr(self.feedforward, "input_channels", None)
        ff_out_ch = getattr(self.feedforward, "output_channels", None)
        fb_in_ch = getattr(self.feedback, "input_channels", None)
        fb_out_ch = getattr(self.feedback, "output_channels", None)

        # Check if the input/output channels are compatible
        if ff_in_ch is None:
            raise ValueError(
                f"The feedforward pass does not possess the attribute input_channels."
            )
        if ff_out_ch is None:
            raise ValueError(
                f"The feedforward pass does not possess the attribute output_channels."
            )
        if fb_in_ch is None:
            raise ValueError(
                f"The feedback pass does not possess the attribute input_channels."
            )
        if fb_out_ch is None:
            raise ValueError(
                f"The feedback pass does not possess the attribute output_channels."
            )

        assert (
            ff_out_ch == fb_in_ch
        ), f"Feedforward pass has {ff_out_ch} output channels, but feedback pass has {fb_in_ch} input channels. They must be the same."
        assert (
            fb_out_ch == ff_in_ch
        ), f"Feedforward pass {ff_in_ch} input channels, but the feedback pass has {fb_out_ch} output channels. They must be the same."

        return ff_in_ch, ff_out_ch

# ============================= RECURSION ================================


class Parallel(nn.Module):
    r"""
    Parallel processing of two input branches. Inherits from :class:`nn.Module`.
    The branches, if are given as a :class:`nn.Module`, :class:`nn.Sequential`, or :class:`OrderedDict`,
    they are converted to a :class:`Series` instance.

    Shape:
        - input: :math:`(B, M, N_{in}, ...)`
        - output: :math:`(B, M, N_{out}, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    :math:`N_{in}` is the number of input channels, and :math:`N_{out}` is the sum of the number of output channels for branch A and B :math:`N_{out} = N_{out}^A + N_{out}^B`.
    Ellipsis :math:`(...)` represents additional dimensions.

        **Arguments**:
            - **brA**: Branch A with size (M, N_{in}, N_{out}^A).
            - **brB**: Branch B with size (M, N_{in}, N_{out}^B).
            - **alias_decay_db** (float, optional): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples. Defaults to None.
            - **sum_output** (bool, optional): If True, the output of the two branches is summed. If False, the output is concatenated. Defaults to True.

        **Attributes**:
            - **branchA** (nn.Module | Series): The feedforward path.
            - **branchB** (nn.Module | Series): The feedback path.
            - **nfft** (int): The number of frequency points.
            - **alias_decay_db** (float): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples.

    """

    def __init__(
        self,
        brA: nn.Module | nn.Sequential | OrderedDict | Series,
        brB: nn.Module | nn.Sequential | OrderedDict | Series,
        sum_output: bool = True,
    ):

        nn.Module.__init__(self)

        # Prepare the feedforward and feedback paths
        if isinstance(brA, (nn.Sequential, OrderedDict)) and not isinstance(brA, Series):
            self.branchA = Series(brA)
            warnings.warn(
                "Branch A has been converted to a Series class instance."
            )
        else:
            self.branchA = brA
        if isinstance(brB, (nn.Sequential, OrderedDict)) and not isinstance(brB, Series):
            self.branchB = Series(brB)
            warnings.warn(
                "Branch B has been converted to a Series class instance."
            )
        else:
            self.branchB = brB

        self.sum_output = sum_output
        # Check nfft and time anti-aliasing decay-envelope parameter values
        self.nfft = self.__check_attribute("nfft")
        self.alias_decay_db = self.__check_attribute("alias_decay_db")

        # Check I/O compatibility
        self.input_channels, self.output_channels = self.__check_io()

    def forward(self,  X: torch.Tensor, ext_param: dict = None):
        r"""
        Applies the closed-loop transfer function to the input tensor X.

            **Arguments**:
                **X** (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.

            **Returns**:
                torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        ext_param_brA = None
        ext_param_brB = None
        if ext_param is not None:
            for key, param in ext_param.items():
                # check if the key is in param_dict
                if 'branchA' in key:
                    ext_param_brA = param
                elif 'branchB' in key:
                    ext_param_brB = param

        YA = self.branchA(X, ext_param_brA)

        YB = self.branchB(X, ext_param_brB)

        if self.sum_output:
            return YA + YB
        else:
            return torch.cat((YA, YB), dim=2)

    # ---------------------- Check methods ----------------------
    def __check_attribute(self, attr: str) -> int | float:
        r"""
        Checks if the branches have the same value of the requested attribute.

            **Arguments**:
                **attr** (str): The attribute to check.

            **Returns**:
                int | float: The attribute value.

            **Raises**:
                ValueError: The two paths have different values of the requested attribute.
        """
        brA_attr = getattr(self.branchA, attr, None)
        if brA_attr is None:
            warnings.warn(
                f"The feedforward pass does not possess the attribute {attr}."
            )
        brB_attr = getattr(self.branchB, attr, None)
        if brB_attr is None:
            warnings.warn(f"The feedback pass does not possess the attribute {attr}.")
        # Then, check that the two paths have the same value of the attribute.
        if brA_attr is not None and brB_attr is not None:
            assert (
                brA_attr == brB_attr
            ), f"Branch A has {attr} = {brA_attr} and branch B has {attr} = {brB_attr}. They must have the same value."

        # Return the attribute value
        attr_value = brA_attr if brA_attr is not None else brB_attr

        return attr_value

    def __check_io(self) -> tuple:
        r"""
        Checks if branch A and branch B have compatible input/output shapes.

            **Returns**:
                tuple(int,int): The number of input and output channels.

            **Raises**:
                - ValueError: If any of the branches does not possess either the input_channels or the output_channels attributes.
                - AssertionError: Branch A and branch B input channels are not compatible.
        """
        # Get input channels of both feedforward and feedback
        brA_in_ch = getattr(self.branchA, "input_channels", None)
        brA_out_ch = getattr(self.branchA, "output_channels", None)
        brB_in_ch = getattr(self.branchB, "input_channels", None)
        brB_out_ch = getattr(self.branchB, "output_channels", None)

        # Check if the input/output channels are compatible
        if brA_in_ch is None:
            raise ValueError(
                f"Branch A does not possess the attribute input_channels."
            )
        if brA_out_ch is None:
            raise ValueError(
                f"Branch A does not possess the attribute output_channels."
            )
        if brB_in_ch is None:
            raise ValueError(
                f"Branch B does not possess the attribute input_channels."
            )
        if brB_out_ch is None:
            raise ValueError(
                f"Branch B does not possess the attribute output_channels."
            )

        assert (
            brA_in_ch == brB_in_ch
        ), f"Branch A has {brA_in_ch} input channels, but branch B has {brB_in_ch} input channels. They must be the same."
        if self.sum_output: 
            assert (
                brA_out_ch == brB_out_ch
            ), f"Branch A has {brA_out_ch} output channels, but branch B has {brB_out_ch} output channels. They must be the same if their output is being summed."
            return brA_in_ch, brA_out_ch
        else:
            return brA_in_ch, brA_out_ch + brB_out_ch
    

# ============================= SHELL ================================


class Shell(nn.Module):
    r"""
    DSP wrapper class. Interfaces the DSP with dataset and loss function. Inherits from :class:`nn.Module`.

    Shape:
        - input: :math:`(B, M, N_{in}, ...)`
        - output: :math:`(B, M, N_{out}, ...)`

    where :math:`B` is the batch size, :math:`M` is the number of frequency bins,
    :math:`N_{in}` is the number of input channels (defined by the `core` and the `input_layer`),
    and :math:`N_{out}` is the number of output channels (defined by the `core` and the `output_layer`).
    Ellipsis :math:`(...)` represents additional dimensions.

        **Arguments / Attributes**:
            - **core** (nn.Module | nn.Sequential): DSP.
            - **input_layer** (nn.Module, optional): layer preceeding the DSP and correctly preparing the Dataset input before the DSP processing. Default: Transform(lambda x: x).
            - **output_layer** (nn.Module, optional): layer following the DSP and preparing its output for the comparison with the Dataset target. Default: Transform(lambda x: x).

        **Attributes**:
            - **nfft** (int): Number of frequency points.
            - **alias_decay_db** (float): The decaying factor in dB for the time anti-aliasing envelope. The decay refers to the attenuation after nfft samples.

    """

    def __init__(
        self,
        core: nn.Module | Recursion | nn.Sequential,
        input_layer: Recursion | Series | nn.Module = nn.Identity(),
        output_layer: Recursion | Series | nn.Module = nn.Identity(),
    ):

        nn.Module.__init__(self)

        # Prepare the core, input layer, and output layer
        if isinstance(core, (nn.Sequential, OrderedDict)) and not isinstance(
            core, Series
        ):
            self.__core = Series(core)
            warnings.warn("Core has been converted to a Series class instance.")
        else:
            self.__core = core
        if isinstance(input_layer, (nn.Sequential, OrderedDict)) and not isinstance(
            input_layer, Series
        ):
            self.__input_layer = Series(input_layer)
            warnings.warn("Input layer has been converted to a Series class instance.")
        else:
            self.__input_layer = input_layer
        if isinstance(output_layer, (nn.Sequential, OrderedDict)) and not isinstance(
            output_layer, Series
        ):
            self.__output_layer = Series(output_layer)
            warnings.warn("Output layer has been converted to a Series class instance.")
        else:
            self.__output_layer = output_layer

        # Check model nfft and time anti-aliasing decay-envelope parameter values
        self.nfft = self.__check_attribute("nfft")
        self.alias_decay_db = self.__check_attribute("alias_decay_db")

        # Check I/O compatibility
        self.input_channels, self.output_channels = self.__check_io()

    def forward(self, x: torch.Tensor, ext_param: dict = None) -> torch.Tensor:
        r"""
        Forward pass through the input layer, the core, and the output layer. Keeps the three components separated.

            **Args**:
                - x (torch.Tensor): Input tensor of shape :math:`(B, M, N_{in}, ...)`.

            **Returns**:
                - torch.Tensor: Output tensor of shape :math:`(B, M, N_{out}, ...)`.
        """
        x = self.__input_layer(x)
        if ext_param is not None:
            x = self.__core(x, ext_param)
        else:
            x = self.__core(x)
        x = self.__output_layer(x)
        return x

    # ---------------------- Get and set methods ----------------------
    def get_inputLayer(self) -> nn.Module | nn.Sequential:
        r"""
        Returns the current input layer.
        """
        return self.__input_layer

    def set_inputLayer(self, input_layer: nn.Module = None) -> None:
        r"""
        Substitutes the current input layer with a given new one.
            **Argument**:
                **input_layer** (nn.Module | nn.Sequential, optional): The new input layer. Defaults to None.
        """
        self.__input_layer = input_layer

    def get_outputLayer(self) -> nn.Module | nn.Sequential:
        r"""
        Returns the current output layer.
        """
        return self.__output_layer

    def set_outputLayer(self, output_layer: nn.Module = None) -> None:
        r"""
        Substitutes the current output layer with a given new one.
            **Argument**:
                **output_layer** (nn.Module | nn.Sequential, optional): The new input layer. Defaults to None.
        """
        self.__output_layer = output_layer

    def get_core(self) -> nn.Module | nn.Sequential:
        r"""
        Returns the current core DSP.
        """
        return self.__core

    def set_core(self, core: nn.Module) -> None:
        r"""
        Substitutes the current core with a given new one.
            **Argument**:
                **output_layer** (nn.Module, optional): The core DSP system. Defaults to None.
        """
        self.__core = core

    # ---------------------- Check methods ----------------------
    def __check_attribute(self, attr: str) -> int | float:
        r"""
        Check if all the modules in core, input layer, and output layer have the same value for the requested attribute.

            **Argument**:
                - **attr** (str): The attribute to check.

            **Returns**:
                int: The attribute value.

            **Raises**:
                - ValueError: The core component does not possess the requested attribute.
                - AssertionError: Core, input layer, and output layer do not have the same value of the requested attribute.
        """

        # Check that core, input layer, and output layer all possess the nfft attribute.
        if getattr(self.__core, attr, None) is None:
            raise ValueError(f"The core does not possess the attribute {attr}.")
        if getattr(self.__input_layer, attr, None) is not None:
            assert getattr(self.__core, attr) == getattr(
                self.__input_layer, attr
            ), f"The input layer has {attr} = {getattr(self.__input_layer, attr)} and the core has {attr} = {getattr(self.__core, attr)}. They must have the same value."
        if getattr(self.__output_layer, attr, None) is not None:
            assert getattr(self.__core, attr) == getattr(
                self.__output_layer, attr
            ), f"The core has {attr} = {getattr(self.__core, attr)} and the output layer has {attr} = {getattr(self.__output_layer, attr)}. They must have the same value."

        # Get current value
        return getattr(self.__core, attr)

    def __check_io(self) -> tuple:
        r"""
        Checks if the input layer, the core, and the output layer have compatible input/output shapes.

            **Returns**:
                tuple(int,int): The number of input and output channels.

            **Raises**:
                ValueError: The core, the input layer, and the output layer are not I/O compatible.
        """
        # Check that core and input layer are I/O compatible
        if getattr(self.__core, "input_channels", None) is None:
            raise ValueError(f"The core does not possess the attribute input_channels.")
        if getattr(self.__input_layer, "output_channels", None) is not None:
            core_in_ch = getattr(self.__core, "input_channels")
            inlayer_out_ch = getattr(self.__input_layer, "output_channels")
            assert (
                core_in_ch == inlayer_out_ch
            ), f"The core should receive {core_in_ch} input channels, but {inlayer_out_ch} channels arrive from the input layer."

        # Check that core and output layer are I/O compatible
        if getattr(self.__core, "output_channels", None) is None:
            raise ValueError(
                f"The core does not possess the attribute output_channels."
            )
        if getattr(self.__output_layer, "input_channels", None) is not None:
            core_out_ch = getattr(self.__core, "output_channels")
            outlayer_in_ch = getattr(self.__output_layer, "input_channels")
            assert (
                core_out_ch == outlayer_in_ch
            ), f"The core sends {core_out_ch} output channels, but the output layer can only receive {outlayer_in_ch} channels."

        # Return the number of input and output channels
        inlayer_in_ch = getattr(self.__input_layer, "input_channels", None)
        outlayer_out_ch = getattr(self.__output_layer, "output_channels", None)

        if inlayer_in_ch is not None:
            in_ch = inlayer_in_ch
        else:
            in_ch = getattr(self.__core, "input_channels")
        if outlayer_out_ch is not None:
            out_ch = outlayer_out_ch
        else:
            out_ch = getattr(self.__core, "output_channels")

        return in_ch, out_ch

    # ---------------------- Responses methods ----------------------
    def get_time_response(
        self, fs: int = 48000, identity: bool = False
    ) -> torch.Tensor:
        r"""
        Generates the impulse response of the DSP.

            **Arguments**:
                - **fs** (int, optional): Sampling frequency. Defaults to 48000.
                - **identity** (bool, optional): If False, return the input-to-output impulse responses of the DSP.
                                        If True, return the input-free impulse responses of the DSP.
                                        Defaults to False.

            **NOTE**: Definition of 'input-to-output' and 'input-free'
                Let :math:`A \in \mathbb{R}^{T \times  N_{out} \times N_{in}}` be a time filter matrix. If :math:`x \in \mathbb{R}^{T \times  N_{in}}` is an :math:`N_{in}`-dimensional time signal having
                a unit impulse at time :math:`t=0` for each element along :math:`N_{in}`. Let :math:`I \in R^{T \times  N \times N}` be an diagonal matrix across
                second and third dimension, with unit impulse at time :math:`t=0`for each element along such diagonal.
                If \* represent the signal-wise matrix convolution operator, then:

                - :math:`y = A * x` is the 'input-to-output' impulse response of :math:`A`.
                - :math:`A * I` is the 'input-free' impulse response of :math:`A`.

            **Returns**:
                - torch.Tensor: Generated DSP impulse response.
        """

        # construct anti aliasing reconstruction envelope
        gamma = 10 ** (-torch.abs(self.alias_decay_db) / (self.nfft) / 20)
        self.alias_envelope = (
            (gamma ** torch.arange(0, -self.nfft, -1, device=gamma.device))
            .view(1, -1, 1)
            .expand(1, -1, self.output_channels)
        )

        # save input/output layers
        input_save = self.get_inputLayer()
        output_save = self.get_outputLayer()

        # update input/output layers
        self.set_inputLayer(FFT(self.nfft))
        self.set_outputLayer(
            nn.Sequential(iFFT(self.nfft), Transform(lambda x: x * self.alias_envelope))
        )

        # generate input signal
        x = signal_gallery(
            batch_size=1,
            n_samples=self.nfft,
            n=self.input_channels,
            signal_type="impulse",
            fs=fs,
            device=gamma.device,
        )
        if identity and self.input_channels > 1:
            self.alias_envelope = self.alias_envelope.unsqueeze(-1).expand(
                1, -1, -1, self.input_channels
            )
            x = x.diag_embed()

        # generate impulse response
        with torch.no_grad():
            y = self.forward(x)

        # restore input/output layers
        self.set_inputLayer(input_save)
        self.set_outputLayer(output_save)

        return y

    def get_freq_response(
        self, fs: int = 48000, identity: bool = False
    ) -> torch.Tensor:
        r"""
        Generates the frequency response of the DSP.

            **Arguments**:
                - **fs** (int, optional): Sampling frequency. Defaults to 48000.
                - **identity** (bool, optional): If False, return the input-to-output frequency responses of the DSP.
                                        If True, return the input-free frequency responses of the DSP.
                                        Defaults to False.

            **NOTE**: Definition of 'input-to-output' and 'input-free'
                Let :math:`A \in \mathbb{R}^{F \times  N_{out} \times N_{in}}` be a frequency filter matrix. If :math:`x \in \mathbb{R}^{T \times  N_{in}}` is an :math:`N_{in}`-dimensional signal having
                a unit impulse at time :math:`t=0` spectrum for each element along :math:`N_{in}`. Let :math:`I \in R^{F \times  N \times N}` be an diagonal matrix across
                second and third dimension, with unit impulse at time :math:`t=0` spectra for each element along such diagonal.
                If \* represent the signal-wise matrix product operator, then:

                - :math:`y = A * x` is the 'input-to-output' frequency response of :math:`A`.
                - :math:`A * I` is the 'input-free' frequency response of :math:`A`.

            **Returns**:
                torch.Tensor: Generated DSP frequency response.
        """

        # contruct anti aliasing reconstruction envelope
        gamma = 10 ** (-torch.abs(self.alias_decay_db) / (self.nfft) / 20)
        self.alias_envelope_exp = (
            (gamma ** torch.arange(0, -self.nfft, -1, device=gamma.device))
            .view(1, -1, 1)
            .expand(1, -1, self.output_channels)
        )

        # save input/output layers
        input_save = self.get_inputLayer()
        output_save = self.get_outputLayer()

        # update input/output layers
        self.set_inputLayer(FFT(self.nfft))
        self.set_outputLayer(
            nn.Sequential(
                iFFT(self.nfft),
                Transform(
                    lambda x: torch.einsum(
                        "bfm..., bfm... -> bfm...", x, self.alias_envelope_exp
                    )
                ),
                FFT(self.nfft),
            )
        )  # TODO, this is a very suboptimal way to do this, we need to find a better way

        # generate input signal
        x = signal_gallery(
            batch_size=1,
            n_samples=self.nfft,
            n=self.input_channels,
            signal_type="impulse",
            fs=fs,
            device=gamma.device,
        )
        if identity and self.input_channels > 1:
            x = x.diag_embed()

        # generate frequency response
        with torch.no_grad():
            y = self.forward(x)

        # restore input/output layers
        self.set_inputLayer(input_save)
        self.set_outputLayer(output_save)

        return y

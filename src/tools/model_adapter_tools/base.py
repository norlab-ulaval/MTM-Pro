# coding=utf-8

from functools import partial
from typing import Any, Callable, Dict, Tuple, Type, Union
import abc

import numpy as np
import torch


class ModelInputOutputAdapterBase:
    # ToDo: RLRP-403 feat: add FeatureDataclass support to ModelInputOutputAdapterBase

    def __init__(
        self, array_in_len: int, array_out_len: int, adapter_fct: Callable, *args, **kwargs
    ):
        """
        This class serves as a foundation for implementing custom model adapters which follow a
        given prototype specification. Those types of adapter provide callable interfaces for
        mapping model input/output to/from environment.

        It validates the adapter's function signature and its output against the specified
        input/output schema, ensuring proper configuration for deployment. The adapter
        functionality is defined by the user-supplied callable `adapter_fct`, which processes
        input data and produces output aligned with the specified dimensions. Subclasses are
        required to specify the model adapter prototype by overiding `_adapter_output_type` and
        `_set_adapter_kwarg_attribute` methods.

        Attribute `adapter_fct` expect the following function signature at the minimum:

            obs_adapter_fct(in_arg: Union[np.ndarray,torch.Tensor],
                            ) -> Union[np.ndarray,torch.Tensor]

        but subclass can be implemented to expect arbitrary input output. For example

            obs_adapter_fct( in_arg: Union[np.ndarray,torch.Tensor],
                             last_env_state: Union[None,Any],
                            ) -> Tuple[Tuple[Type[np.ndarray], Type[torch.Tensor]], Type[Dict]]

        See `tools/model_adapter_tools/deployer_adapter.py` subclassing examples.

        Notes:

        - Can handle both numpy ndarray and pytorch tensor
        - Can handle multidimensional obs array with ensemble, batch and feature dimensions. Assume
          input/output array shape len > 1 corresspond to `( E, ... , nDim )` with the ensemble
          dimension `E` and feature dimension `nDim`.

        :param array_in_len: Length of the input observation array.
        :param array_out_len: Length of the output observation array.
        :param adapter_fct: Callable function used to transform the observation array.
        """

        self.adapter_fct = adapter_fct
        self.array_out_len = array_out_len
        self.array_in_len = array_in_len
        self.adapter_kwarg_attribute = {}

        self._array_input_mock = np.zeros((self.array_in_len,))
        self._array_output_mock = np.zeros((self.array_out_len,))

        self._set_adapter_kwarg_attribute()
        self._validate_adapter_fct_signature()
        self._validate_adapter_output()

        self._set_adapter_for_deployment()

    @abc.abstractmethod
    def _adapter_output_type(self) -> Union[
        Tuple[Type[np.ndarray], Type[torch.Tensor]],
        Tuple[Tuple[Type[np.ndarray], Type[torch.Tensor]], Any],
    ]:
        """
        Specify the `adapter_fct` output type.

        This method determines the expected output type(s) of an adapter function, providing
        flexibility for different use cases. It enforces subclasses to define and return the
        specific data types or a tuple containing them. The return types can include array data
        structures and potentially other compound types to cater to various adaptation needs.

        At the minimum, it must return `Tuple[Type[np.ndarray], Type[torch.Tensor]]`, e.g.

            >>> def _adapter_output_type(self):
            >>>     return np.ndarray, torch.Tensor

        or `Tuple[Tuple[Type[np.ndarray], Type[torch.Tensor]], Any]`, e.g.:

            >>> def _adapter_output_type(self):
            >>>     return ((np.ndarray, torch.Tensor), dict)

        :return: A tuple that specifies the adapter output types, which can include numpy ndarrays,
            PyTorch tensors, or a nested combination of these types with other possible elements.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _set_adapter_kwarg_attribute(self) -> None:
        """
        Sets an adapter keyword argument attribute for the current instance.

        Usage example:

            >>> def _set_adapter_kwarg_attribute(self) -> None:
            >>>     self.adapter_kwarg_attribute["<model-adapter-kwarg>"] = <value>
            >>>     return None

        The method modifies or initializes an adapter-specific keyword argument
        attribute in the instance. The exact behavior and usage depend on specific
        class and context. This method does not return any value.

        :return: None
        """
        raise NotImplementedError

    def _validate_adapter_output(self):
        output: Union[
            Union[np.ndarray, torch.Tensor],
            Tuple[Union[np.ndarray, torch.Tensor], Any],
        ] = self.adapter_fct(self._array_input_mock, **self.adapter_kwarg_attribute)

        if self._adapter_output_type() == (np.ndarray, torch.Tensor):
            # Case expected output type: Union[np.ndarray,torch.Tensor]
            assert isinstance(output, self._adapter_output_type()), (
                "The adapter expected output type does not match the adapter real output type "
                f"exp={self._adapter_output_type()} != real={type(output)}"
            )
            array_output = output
        else:
            # Case expected output type: Tuple[Tuple[Type[np.ndarray], Type[torch.Tensor]], Any]
            assert isinstance(output, tuple), (
                "The adapter expected output type does not match the adapter real output type "
                f"real={type(output)} not in exp={self._adapter_output_type()}"
            )
            for idx in np.arange(len(output)):
                assert isinstance(output[idx], self._adapter_output_type()[idx]), (
                    "The adapter expected output type does not match the adapter real output "
                    f"type real={type(output[idx])} not in exp="
                    f"{self._adapter_output_type()[idx]}"
                )
            array_output = output[0]

        assert self._array_output_mock.shape == array_output.shape, (
            "The adapter obs array expected output does not match the adapter obs array real "
            f"output exp={self._array_output_mock.shape} != real={array_output.shape}"
        )

    def _validate_adapter_fct_signature(self) -> None:
        try:
            self.adapter_fct(self._array_input_mock, **self.adapter_kwarg_attribute)
        except TypeError as e:
            raise AttributeError(
                f"Adapter {self.adapter_fct} is missing required keyword "
                f"argument in its function signature \n{e}"
            )
        return None

    def _set_adapter_for_deployment(self):
        self.adapter_fct = partial(self.adapter_fct, **self.adapter_kwarg_attribute)

    def __call__(self, in_arg: Union[np.ndarray, torch.Tensor], *args, **kwargs) -> Union[
        Union[np.ndarray, torch.Tensor],
        Tuple[Union[np.ndarray, torch.Tensor], Any],
    ]:
        """Execute observation adapter callable.

        Assume input/output array shape len > 1 corresspond to `( E, ... , nDim )`
        with the ensemble dimension `E` and feature dimension `nDim`.

        Note: handle model ensemble dimension by transposing the input and re-transposing back
        the ouput.
        """
        if isinstance(in_arg, torch.Tensor):
            transposed_input = torch.transpose(in_arg, 0, -1)
            adapted_data = self.adapter_fct(transposed_input, *args, **kwargs)
            if self._adapter_output_type() == (np.ndarray, torch.Tensor):
                return torch.transpose(adapted_data, 0, -1)
            else:
                adapted_data_array, adapted_data = adapted_data
                return torch.transpose(adapted_data_array, 0, -1), adapted_data
        elif isinstance(in_arg, np.ndarray):
            if in_arg.ndim == 1:
                return self.adapter_fct(in_arg, *args, **kwargs)
            else:
                if self._adapter_output_type() == (np.ndarray, torch.Tensor):
                    return self.adapter_fct(in_arg.T, *args, **kwargs).T
                else:
                    adapted_data_array, adapted_data = self.adapter_fct(in_arg.T, *args, **kwargs)
                    return adapted_data_array.T, adapted_data
        else:
            raise NotImplementedError(f"input type {type(in_arg)} not suported")

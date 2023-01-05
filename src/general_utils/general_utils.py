import time

from beartype import beartype
import collections
import contextlib
import functools
import h5py  # type: ignore[import]
import inspect
import itertools
import json
import logging
import math
import natsort
import numpy as np
import os
from pathlib import Path
import rich
import subprocess
import types; 
import torch
from typing import *


def log_rank_0(logger, level, message):
    if os.getenv("RANK", "0") == "0":
        logger.log(level, "[white bold]\[log-zero]:[/] " + message)


def info_rank_0(logger, message):
    log_rank_0(logger, logging.INFO, message)


def debug_rank_0(logger, message):
    log_rank_0(logger, logging.DEBUG, message)



def color_cycle(text, colors_fg=None, colors_bg=None):
    assert colors_fg or colors_bg, "Must specify at least one color."
    
    output = []
    
    color_it = iter(zip(
        itertools.cycle(colors_fg) if colors_fg else itertools.repeat(""),
        itertools.cycle(colors_bg) if colors_bg else itertools.repeat(""),
    ))
    
    for char in text:
        fg, bg = next(color_it)
        bg = f"on {bg}" if bg else ""
        # Appending to a string repetitively has bad algorithmic complexity, because a new string is created each time.
        # We use a list instead.
        output.append(f"[{fg}{bg}]{char}[/{fg}{bg}]")
    
    return "".join(output)


def global_rank() -> int:
    return int(os.environ.get("SLURM_PROCID", 0))


def is_rank_zero():
    return global_rank() == 0


def rich_print_zero_rank(*args, **kwargs):
    if is_rank_zero():
        rich.print(*args, **kwargs)

@contextlib.contextmanager
def cuda_timeit(name, disable=False):
    if disable:
        yield
        return

    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    rich.print(f"\n[bold blue]{name}[/] took {end - start:0.5f} seconds")


@contextlib.contextmanager
def ctx_timeit(name, disable=False):
    if disable:
        yield
        return

    start = time.perf_counter()
    yield
    end = time.perf_counter()
    rich.print(f"\n[bold blue]{name}[/] took {end - start:0.5f} seconds")


###############################################################################
# Checks
###############################################################################
def check_in(value, container):
    assert value in container, f"\"{value}\" not in {container}"

check_contained = check_in


def check_equal(a, b):
    assert a == b, f"{a} != {b}"

    
def check_isinstance(obj, types):
    assert isinstance(obj, types), f"{type(obj).mro()} is not a subclass of {types}"


def check_not_equal(a, b):
    assert a != b, f"{a} == {b}"
    

def check_and_print_args(all_arguments, function, has_classmethod_cls=False, root_path=None):
    check_args(all_arguments, function, has_classmethod_cls)
    rich.print("[bold]Arguments:")
    print_dict(all_arguments, root_path)
    print()


def check_args(all_arguments, function, has_classmethod_cls=False):
    """
    We get the arguments by calling `locals`. This makes sure that we
    really called locals at the very beginning of the function, otherwise
    we have supplementary keys.

    There is a weird edge-case that classmethods' `cls` argument is
    not returned by inspect.signature, so we have to add it.
    """

    inspect_args = set(inspect.signature(function).parameters.keys())
    if has_classmethod_cls:
        if isinstance(has_classmethod_cls, str):
            inspect_args.add(has_classmethod_cls)
        else:
            inspect_args.add("cls")

    assert (all_arguments.keys() == inspect_args), (
        f"\n{sorted(all_arguments.keys())} != "
        f"{sorted(inspect.signature(function).parameters.keys())}"
    )


def check_shape(shape: Sequence[int], expected_shape: Sequence[int]):
    if not shape == expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {shape}.")


###############################################################################
# Print utils
###############################################################################

def print_list(_list, root_path=None):
    at_least_one = False
    for line in _list:
        at_least_one = True
        if root_path and isinstance(line, (str, Path)):
            line = shorten_path(line, root_path)
        rich.print(f"\t- {line}")

    if not at_least_one:
        rich.print("\t[bright_black]<empty list>")


def print_dict(
    _dict: dict[str, Any], 
    root_path=None, 
    return_str=False,
) -> None:
    # Pad by key length
    max_len = len(max(_dict, key=lambda key: len(str(key)))) + 1
    at_least_one = False

    if return_str:
        output = []

    for k, value in _dict.items():
        at_least_one = True
        if root_path and isinstance(value, (Path, str)):
            value = shorten_path(value, root_path)

        text = (
            f"\t- [white bold]{k}[/] " + 
            (max_len - len(k)) * " " + 
            f" = [green]{value}"
        )

        if return_str:
            output.append(text)
        else:
            rich.print(text)

    if not return_str:
        if not at_least_one:
            rich.print("\t[bright_black]<empty dict>")

    if return_str:
        return "\n".join(output)


def clean_locals(locals_, no_modules=True, no_classes=True, no_functions=True, no_caps=True):
    print([
        k for k, v in locals_.items() if 
        not k.startswith("_") and 
        (not no_modules or not isinstance(v, types.ModuleType)) and 
        (not no_classes or not isinstance(v, type)) and 
        (not no_functions or not callable(v)) and 
        (not no_caps or k == k.lower())
    ])


@beartype
def shorten_path(path: Union[Path, str], root_node: Union[Path, str]) -> str:
    
    if isinstance(path, str):
        path = Path(path.strip())

    if path.is_relative_to(root_node):
        path = "<pwd> /" + str(path.relative_to(root_node))
    else:
        path = str(path)
    return path


_to_human_size_SIZE_HUMAN_NAMES = {
        0: "B",
        1: "KB",
        2: "MB",
        3: "GB",
    }

def to_human_size(size: int) -> str:
    if size == 0:
        return "0 B"

    exponent = int(math.log(size, 1000))
    mantissa = size / 1000 ** exponent
    return f"{mantissa:.2f} {_to_human_size_SIZE_HUMAN_NAMES[exponent]}"


###############################################################################
# Common collection manipulations, including strings and iterators
###############################################################################
def repeat_interleave(iterable, n: int):
    return itertools.chain.from_iterable(
        itertools.repeat(x, n) for x in iterable)


def safe_xor(a, b):
    return bool(a) ^ bool(b)
    

def setattr_must_exist(obj, key, value):
    assert hasattr(obj, key), f"Key `{key}` does not exist."
    setattr(obj, key, value)


def dict_assign_must_exist(d, key, value):
    assert key in d, f"Key `{key}` does not exist."
    d[key] = value


def dict_zip(dict_of_lists: dict[Any, Iterable[Any]], check_lengths=True
) -> Generator[dict[Any, Any], None, None]:
    """
    Takes a dict of iterables and returns a generator of dicts 
    with the same keys and a value of the lists at each iteration.
    """

    if check_lengths:
        # Requires all lists to have the same length.
        # Could do some iterator stuff to keep track of the length. Haven't done that yet.
        l0 = len(next(iter(dict_of_lists.values())))
        assert all(l0 == len(v) for v in dict_of_lists.values()
        ), {k: len(v) for k, v in dict_of_lists.items()}

    iters = {k: iter(v) for k, v in dict_of_lists.items()}
    
    while True:
        try:
            yield {k: next(iters[k]) for k in dict_of_lists.keys()}

        except StopIteration:
            break


def dict_unzip(list_of_dicts: list[dict[Any, Any]], key_subset=None) -> dict[Any, list[Any]]:
    """
    Unzips a list of dicts with the same keys into a dict of lists.
    """
    if key_subset is None:
        keys = list_of_dicts[0].keys()
    else:
        keys = key_subset

    dict_of_lists = collections.defaultdict(list)
    
    for ld in list_of_dicts:
        assert ld.keys() == keys, f"{ld.keys()} != {keys}"
        for k in keys:
            dict_of_lists[k].append(ld[k])

    return dict_of_lists


def concat_lists(lists):
    assert all(isinstance(l, list) for l in lists)
    return sum(lists, [])


def concat_tuples(tuples):
    assert all(isinstance(l, tuple) for l in tuples)
    return sum(tuples, ())


def concat_iters(iters):
    return list(itertools.chain.from_iterable(iters))


def only_one(it: Iterable):
    iterated = iter(it)
    good = next(iterated)
    for bad in iterated:
        raise ValueError("Expected only one item, got more than one.")
    return good


def find_last(find_in: Sequence[Any], find_what: Any) -> int:
    return len(find_in) - find_in[::-1].index(find_what) - 1


def sort_iterable_text(list_text):
    return natsort.natsorted(list_text)


###############################################################################
# Common file and process manipulations
###############################################################################
def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
load_json = read_json


def write_json(data: dict, path: Path, exists_ok=True, **kwargs) -> None:
    if not exists_ok and path.exists():
        raise FileExistsError(f"{path} already exists")

    with open(path, "w") as f:
        json.dump(data, f, indent=4, **kwargs)
dump_json = write_json


def cmd(command: list[str]) -> list[str]:
    return subprocess.check_output(command).decode().strip().split("\n")


def count_lines(path: Path) -> int:
    return int(check_len(only_one(cmd(["wc", "-l", str(path)])).split(), 2)[0])


def check_len(seq: Sequence, expected_len: int) -> Sequence:
    if not len(seq) == expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(seq)}.")
    return seq


_print_structure_h5_H5_POSSIBLE_TYPES = (h5py.Dataset, h5py.Group, h5py.File)

def print_structure_h5(file_object: h5py.File):
    work_stack = [(file_object, "")]
    all_text = []

    while work_stack:

        obj, parent_name = work_stack.pop()
        assert isinstance(
            obj, _print_structure_h5_H5_POSSIBLE_TYPES)

        if obj.name == "":
            obj_name = "<root>"
        else:
            obj_name = obj.name
        
        if parent_name:    
            name = parent_name + "/" + obj_name
        else:
            name = obj_name

        message = f"\"{name}\": {type(obj).__name__}"
        if isinstance(obj, h5py.Dataset):
            message += f" {obj.shape} {obj.dtype}"
        all_text.append(message)

        if obj.attrs:
            for k, v in obj.attrs.items():
                message_attr = f"\t- {k}: {type(v).__name__}"
                if isinstance(v, (str, int, float)):
                    message_attr += f" value=\"{v}\""
                elif isinstance(v, np.ndarray):
                    message_attr += f" shape={v.shape} dtype={v.dtype}"
                elif isinstance(v, (tuple, list, dict)):
                    message_attr += f" {type(v)} {len(v)}"
                all_text.append(message_attr)

        if hasattr(obj, "items"):
            assert isinstance(obj, (h5py.Group, h5py.File))
            for key, value in obj.items():
                work_stack.append((value, parent_name))

    rich.print("\n".join(all_text))
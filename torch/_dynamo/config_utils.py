import contextlib
import copy

import dataclasses
import inspect
import os

import pickle

import tempfile
import unittest
from os.path import abspath, dirname
from types import FunctionType, ModuleType
from typing import Any, Dict, Optional, Set, Type
from unittest import mock

import torch

from . import external_utils






def patch_object(obj, name, value):
    """
    Workaround `mock.patch.object` issue with ConfigModule
    """
    if isinstance(obj, ConfigMixin):
        return obj.patch(name, value)
    return mock.patch.object(obj, name, value)

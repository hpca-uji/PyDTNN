# Typing related (or non important) imports
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from inspect import stack # This is only in order to get the function's name

# Functionality imports
# EMPTY (for now)

def Binarizer(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Binarizer --- #

def CastMap(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END CastMap --- #

def CategoryMapper(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END CategoryMapper --- #

def DictVectorizer(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END DictVectorizer --- #

def FeatureVectorizer(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END FeatureVectorizer --- #

def Imputer(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Imputer --- #

def LabelEncoder(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END LabelEncoder --- #

def LinearClassifier(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END LinearClassifier --- #

def LinearRegressor(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END LinearRegressor --- #

def Normalizer(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Normalizer --- #

def OneHotEncoder(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END OneHotEncoder --- #

def SVMClassifier(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END SVMClassifier --- #

def SVMRegressor(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END SVMRegressor --- #

def Scaler(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END Scaler --- #

def TreeEnsemble(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END TreeEnsemble --- #

def TreeEnsembleClassifier(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END TreeEnsembleClassifier --- #

def TreeEnsembleRegressor(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END TreeEnsembleRegressor --- #

def ZipMap(info: Dict[str, Any]) -> LayerAndActivationBase:
    print(f"{stack()[0].function} args received: {info}")
    raise NotImplementedError("Not implemented")
# --- END ZipMap --- #

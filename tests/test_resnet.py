import pytest
from tensorflow.keras import backend as K
from resnet import ResnetBuilder


DIM_ORDERING = {'th', 'tf'}


def _test_model_compile(model):
    for ordering in DIM_ORDERING:
        #K.set_image_dim_ordering(ordering)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to compile with '{}' dim ordering".format(ordering)


def test_resnet18():
    #model = ResnetBuilder.build_resnet_18((3, 224, 224), 100)
    model = ResnetBuilder.build_resnet_18((224, 224, 3), 100)
    _test_model_compile(model)


def test_resnet34():
    model = ResnetBuilder.build_resnet_34((224, 224, 3), 100)
    _test_model_compile(model)


def test_resnet50():
    model = ResnetBuilder.build_resnet_50((224, 224, 3), 100)
    _test_model_compile(model)


def test_resnet101():
    model = ResnetBuilder.build_resnet_101((224, 224, 3), 100)
    _test_model_compile(model)


def test_resnet152():
    model = ResnetBuilder.build_resnet_152((224, 224, 3), 100)
    _test_model_compile(model)


def test_custom1():
    """ https://github.com/raghakot/keras-resnet/issues/34
    """
    model = ResnetBuilder.build_resnet_152((300, 300, 3), 100)
    _test_model_compile(model)


def test_custom2():
    """ https://github.com/raghakot/keras-resnet/issues/34
    """
    model = ResnetBuilder.build_resnet_152((512, 512, 3), 2)
    _test_model_compile(model)


if __name__ == '__main__':
    pytest.main([__file__])
